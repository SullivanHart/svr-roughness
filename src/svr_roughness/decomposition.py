from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import ndimage
from scipy.spatial import cKDTree

from .algorithm import analyze_pure_python
from .config import RoughnessConfig
from .io import load_points
from .result import RoughnessResult, format_report


@dataclass(frozen=True)
class DecompositionConfig:
    """Configuration for 3D multi-patch surface decomposition."""

    plane_distance_thresh_mm: float = 1.5
    edge_margin_mm: float = 3.0  # expansion radius around curvature-flagged edge seeds
    edge_neighbor_radius_mm: float = 4.5
    edge_neighbor_count: int = 120
    edge_curvature_threshold: float = 0.02
    cluster_cell_mm: float = 2.0
    min_patch_points: int = 5000
    min_patch_area_mm2: float = 400.0
    astm_min_area_mm2: float = 2500.0  # ASTM WK92969 §3.1.5 recommends >= 50x50 mm
    max_faces: int = 12
    ransac_iterations: int = 250
    subsample_target: int = 120000
    parallel_angle_thresh_deg: float = 25.0
    flatness_ratio_thresh: float = 0.06
    min_plane_separation_mm: float = 10.0


@dataclass
class SurfacePatchResult:
    """ASTM metrology results for an individual segmented 3D planar surface patch."""

    patch_id: int
    name: str
    normal: tuple[float, float, float]
    centroid: tuple[float, float, float]
    area_mm2: float
    dims_mm: tuple[float, float]
    point_count: int
    is_astm_compliant: bool
    roughness: RoughnessResult
    points: np.ndarray = field(repr=False)
    invalid_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)), repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert patch result to clean JSON-serializable dictionary."""
        return {
            "patch_id": self.patch_id,
            "name": self.name,
            "normal": [round(float(x), 4) for x in self.normal],
            "centroid": [round(float(x), 2) for x in self.centroid],
            "area_mm2": round(float(self.area_mm2), 1),
            "dims_mm": [round(float(x), 1) for x in self.dims_mm],
            "point_count": int(self.point_count),
            "is_astm_compliant": bool(self.is_astm_compliant),
            "roughness": self.roughness.to_dict(),
        }


@dataclass
class ObjectRoughnessResult:
    """Aggregate multi-surface ASTM metrology result for a complete 3D object."""

    patches: list[SurfacePatchResult]
    total_points: int
    assigned_points: int
    unassigned_points: int
    coverage_pct: float
    mean_svr_um: float
    worst_svr_um: float
    best_svr_um: float
    dominant_patch: SurfacePatchResult
    config: RoughnessConfig
    decomp_config: DecompositionConfig
    unassigned_points_arr: np.ndarray | None = field(default=None, repr=False)

    @property
    def patch_count(self) -> int:
        return len(self.patches)

    def to_dict(self) -> dict[str, Any]:
        """Convert object result to clean JSON-serializable dictionary."""
        return {
            "patch_count": self.patch_count,
            "total_points": int(self.total_points),
            "assigned_points": int(self.assigned_points),
            "unassigned_points": int(self.unassigned_points),
            "coverage_pct": round(float(self.coverage_pct), 1),
            "mean_svr_um": round(float(self.mean_svr_um), 3),
            "worst_svr_um": round(float(self.worst_svr_um), 3),
            "best_svr_um": round(float(self.best_svr_um), 3),
            "dominant_patch_id": self.dominant_patch.patch_id,
            "patches": [p.to_dict() for p in self.patches],
        }

    def format_report(self) -> str:
        """Format a multi-face inspection summary report."""
        lines = [
            "═══════════════════════════════════════════════════════════════",
            "  Multi-Surface 3D Object Roughness Report",
            "═══════════════════════════════════════════════════════════════",
            "",
            f"  Total Scan Points:       {self.total_points:,}",
            f"  Assigned Face Points:    {self.assigned_points:,} ({self.coverage_pct:.1f}% surface coverage)",
            f"  Detected Surface Faces:  {self.patch_count}",
            "",
            "───────────────────────────────────────────────────────────────",
            "  Global 3D Object Roughness Summary",
            "───────────────────────────────────────────────────────────────",
            f"  Area-Weighted Mean S_VR: {self.mean_svr_um:.3f} µm",
            f"  Worst-Case Face S_VR:    {self.worst_svr_um:.3f} µm  (Primary Spec Gating)",
            f"  Best-Case Face S_VR:     {self.best_svr_um:.3f} µm",
            "",
            "───────────────────────────────────────────────────────────────",
            "  Individual Detected Surface Faces",
            "───────────────────────────────────────────────────────────────",
        ]
        for p in self.patches:
            r = p.roughness
            comp = r.comparator_equivalents()
            scrata = comp.get("SCRATA (A802)", comp.get("SCRATA (ASTM A802)", "N/A"))
            status = "PASS (>=50x50mm)" if p.is_astm_compliant else "WARN (<50x50mm)"
            lines.extend(
                [
                    f"  [{p.name}]",
                    f"    Points:     {p.point_count:,} ({p.dims_mm[0]:.1f} × {p.dims_mm[1]:.1f} mm, Area: {p.area_mm2:.0f} mm²)",
                    f"    Face Size:  {status}",
                    f"    S_VR:       {r.svr_um:.3f} µm  |  Sa: {r.sa_um:.3f} µm  |  Sq: {r.sq_um:.3f} µm",
                    f"    SCRATA:     {scrata}",
                    "",
                ]
            )
        lines.append("═══════════════════════════════════════════════════════════════")
        return "\n".join(lines)


def is_planar_surface(
    points: npt.ArrayLike,
    flatness_ratio_thresh: float = 0.06,
    subsample_max: int = 50000,
) -> bool:
    """Determine whether an Nx3 point cloud represents a single planar surface patch."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 10:
        return True
    if len(pts) > subsample_max:
        pts = pts[:: len(pts) // subsample_max]

    c = np.mean(pts, axis=0)
    centered = pts - c
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    # Eigenvalues proportional to s^2
    eig = s**2
    if eig[0] <= 1e-12:
        return True

    flatness = eig[2] / (eig[0] + eig[1] + eig[2])
    bbox = np.ptp(pts, axis=0)
    sorted_dims = np.sort(bbox)
    thickness_ratio = sorted_dims[0] / max(sorted_dims[2], 1e-6)

    return bool(flatness < flatness_ratio_thresh and thickness_ratio < 0.25)


def _near_face_boundary(points: np.ndarray, radius_mm: float, cell_mm: float = 1.0) -> np.ndarray:
    """Locate the outer contour without treating enclosed scan gaps as part edges."""
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    centered = points - points.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    uv = centered @ axes[:2].T
    cells = np.floor((uv - uv.min(axis=0)) / cell_mm).astype(np.int64)
    shape = cells.max(axis=0) + 1
    if np.prod(shape) > max(4_000_000, len(points) * 100):
        return np.zeros(len(points), dtype=bool)
    occupied = np.zeros(tuple(shape + 2), dtype=bool)
    occupied[cells[:, 0] + 1, cells[:, 1] + 1] = True
    # Close narrow scanner gaps before filling interior holes. The padding keeps
    # the actual exterior outside the scan rather than at the array boundary.
    occupied = ndimage.binary_closing(occupied, structure=np.ones((3, 3), dtype=bool))
    occupied = ndimage.binary_fill_holes(occupied)
    distance = ndimage.distance_transform_edt(occupied) * cell_mm
    return distance[cells[:, 0] + 1, cells[:, 1] + 1] <= radius_mm


def _curvature_edge_mask(
    points: np.ndarray,
    candidates: np.ndarray,
    config: DecompositionConfig,
    tree: cKDTree,
) -> np.ndarray:
    """Flag locally non-planar points via covariance eigenvalue ratio (Bazazian 2015).

    σ = λ_min / (λ_min + λ_mid + λ_max) measures how planar a point's local
    neighborhood is. On a flat surface σ ≈ 0; on an edge/corner σ > 0.01-0.03.

    Unlike the old plane-distance approach, this is geometry-independent: it
    works on cubes, cylinders, fillets, and organic castings because it examines
    the LOCAL shape rather than checking distance to other detected planes.

    After flagging high-curvature seeds, expands a margin around them so the
    transition zone between flat faces is fully excluded.
    """
    if len(candidates) == 0 or config.edge_curvature_threshold <= 0:
        return np.zeros(len(candidates), dtype=bool)

    cand_pts = points[candidates]

    # Casting pits, pores, and rough surface textures in the interior of a face
    # have elevated local curvature but are valid surface data, not part edges.
    # Restricting edge seeds to the boundary zone of the face first skips
    # expensive KD-tree queries and 3x3 eigenvalue decompositions on 90-95% of points.
    boundary = _near_face_boundary(cand_pts, config.edge_margin_mm)
    if not np.any(boundary):
        return np.zeros(len(candidates), dtype=bool)

    boundary_sub_indices = np.where(boundary)[0]
    sub_cands = candidates[boundary_sub_indices]

    # Use full-cloud tree for neighbor lookup so edge points whose neighbors
    # span multiple faces are correctly detected as non-planar.
    k = min(config.edge_neighbor_count, len(points))
    sub_flagged = np.zeros(len(sub_cands), dtype=bool)

    for start in range(0, len(sub_cands), 4096):
        end = min(start + 4096, len(sub_cands))
        batch = sub_cands[start:end]
        _, indices = tree.query(
            points[batch], k=k,
            distance_upper_bound=config.edge_neighbor_radius_mm,
        )
        if indices.ndim == 1:
            indices = indices[:, None]

        usable = indices < len(points)
        counts = usable.sum(axis=1)

        # Gather neighbor coordinates, masking invalid slots
        xyz = points[np.minimum(indices, len(points) - 1)]
        xyz = np.where(usable[..., None], xyz, 0.0)
        mean = xyz.sum(axis=1) / np.maximum(counts[:, None], 1)
        centered = np.where(usable[..., None], xyz - mean[:, None, :], 0.0)

        # Batch covariance → eigenvalues
        covariance = np.einsum('nki,nkj->nij', centered, centered) / np.maximum(
            counts[:, None, None], 1,
        )
        eigenvalues = np.linalg.eigvalsh(covariance)  # sorted ascending
        total_var = eigenvalues.sum(axis=1)
        sigma = eigenvalues[:, 0] / np.maximum(total_var, 1e-12)

        # Flag: must have enough neighbors AND exceed curvature threshold
        sub_flagged[start:end] = (counts >= 8) & (sigma > config.edge_curvature_threshold)

    flagged = np.zeros(len(candidates), dtype=bool)
    flagged[boundary_sub_indices[sub_flagged]] = True

    # ── Expand edge margin around confirmed boundary edge seed points ──
    # This catches the transition zone between flat face and edge/corner
    # that individually might have borderline σ values.
    if np.any(flagged) and config.edge_margin_mm > 0:
        seed_pts = cand_pts[flagged]
        seed_tree = cKDTree(seed_pts)
        distance, _ = seed_tree.query(
            cand_pts,
            distance_upper_bound=config.edge_margin_mm,
        )
        flagged |= (distance <= config.edge_margin_mm)

    return flagged


def _stat_outlier_filter(
    points: np.ndarray,
    k_neighbors: int = 6,
    std_mul: float = 3.0,
) -> np.ndarray:
    """Statistical outlier removal matching C++ StatOutlierRemoval(setMeanK=6, setStddevMulThresh=3.0).

    For each point, computes the mean distance to its K nearest neighbors.
    Points whose mean neighbor distance exceeds (global_mean + std_mul * global_std)
    are flagged as outliers (True = outlier, False = inlier).
    """
    if len(points) < k_neighbors + 1:
        return np.zeros(len(points), dtype=bool)

    tree = cKDTree(points)
    k = min(k_neighbors + 1, len(points))  # +1 because query includes self
    dists, _ = tree.query(points, k=k)

    # Mean distance to K nearest neighbors (exclude self at index 0)
    mean_dists = dists[:, 1:].mean(axis=1)

    global_mean = float(np.mean(mean_dists))
    global_std = float(np.std(mean_dists))
    threshold = global_mean + std_mul * global_std

    return mean_dists > threshold


def _largest_spatial_component(points: np.ndarray, cell_mm: float) -> np.ndarray:
    """Keep the largest connected occupied region in a face's local 2D plane."""
    if len(points) == 0 or cell_mm <= 0:
        return np.ones(len(points), dtype=bool)
    centered = points - points.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    uv = centered @ axes[:2].T
    cells = np.floor((uv - uv.min(axis=0)) / cell_mm).astype(np.int64)
    shape = cells.max(axis=0) + 1
    if np.prod(shape) > max(4_000_000, len(points) * 100):
        return np.ones(len(points), dtype=bool)
    occupied = np.zeros(tuple(shape), dtype=bool)
    occupied[cells[:, 0], cells[:, 1]] = True
    labels, count = ndimage.label(occupied, structure=np.ones((3, 3), dtype=np.int8))
    if count <= 1:
        return np.ones(len(points), dtype=bool)
    point_labels = labels[cells[:, 0], cells[:, 1]]
    sizes = np.bincount(point_labels, minlength=count + 1)
    return point_labels == np.argmax(sizes[1:]) + 1


def decompose_3d_object(
    points_or_file: str | Path | npt.ArrayLike,
    config: RoughnessConfig | None = None,
    decomp_config: DecompositionConfig | None = None,
    progress: Callable[[str, float], None] | None = None,
) -> ObjectRoughnessResult:
    """Decompose a complete 3D object scan into ASTM WK92969 planar surface patches and compute metrology.

    Parameters
    ----------
    points_or_file : str, Path, or Nx3 array
        Point cloud file path (.ply, .pcd, .stl, .obj, .csv) or coordinate array in mm.
    config : RoughnessConfig, optional
        Metrology configuration (grid pitch, cutoffs, noise subtraction).
    decomp_config : DecompositionConfig, optional
        Decomposition parameters (plane distance, edge margins, minimum area).
    progress : callable, optional
        Progress callback taking (status_text, percent_0_to_1).

    Returns
    -------
    ObjectRoughnessResult
        Container holding segmented surface patches and multi-face metrology.
    """
    if config is None:
        config = RoughnessConfig()
    if decomp_config is None:
        decomp_config = DecompositionConfig()

    if progress:
        progress("Loading 3D scan points", 0.05)

    if isinstance(points_or_file, (str, Path)):
        points = load_points(points_or_file)
    else:
        points = np.asarray(points_or_file, dtype=np.float64)

    finite = np.isfinite(points).all(axis=1) if points.ndim == 2 else np.array([])
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected an Nx3 point array")
    points = points[finite]
    total_pts = len(points)
    if total_pts < decomp_config.min_patch_points:
        raise ValueError(f"Insufficient points ({total_pts}) for 3D object decomposition")

    # Fast single-plane check
    if is_planar_surface(points, flatness_ratio_thresh=decomp_config.flatness_ratio_thresh):
        if progress:
            progress("Surface identified as single planar patch", 0.40)
        from .analyze import analyze_points

        connected = _largest_spatial_component(points, decomp_config.cluster_cell_mm)
        connected_idx = np.flatnonzero(connected)
        edge_mask = _curvature_edge_mask(points, connected_idx, decomp_config, cKDTree(points))
        clean_pts = points[connected_idx[~edge_mask]]
        clean_idx = connected_idx[~edge_mask]
        # Statistical outlier removal (matches C++ setMeanK=6, setStddevMulThresh=3.0)
        if len(clean_pts) > 10:
            outlier_mask = _stat_outlier_filter(clean_pts, k_neighbors=6, std_mul=3.0)
            face_points = clean_pts[~outlier_mask]
            stat_outliers = clean_pts[outlier_mask]
        else:
            face_points = clean_pts
            stat_outliers = np.zeros((0, 3), dtype=np.float64)

        if len(face_points) > 10:
            c_tmp = np.mean(face_points, axis=0)
            diff_tmp = face_points - c_tmp
            _, _, v_tmp = np.linalg.svd((diff_tmp.T @ diff_tmp) / len(face_points))
            z_res = np.dot(diff_tmp, v_tmp[2])
            med_z = np.median(z_res)
            mad_z = np.median(np.abs(z_res - med_z))
            z_limit = max(0.40, min(decomp_config.plane_distance_thresh_mm, 4.0 * 1.4826 * mad_z))
            valid_z = np.abs(z_res - med_z) <= z_limit

            if np.any(valid_z):
                u_res = np.dot(diff_tmp, v_tmp[0])
                v_res = np.dot(diff_tmp, v_tmp[1])
                sub_pts = np.column_stack((u_res[valid_z], v_res[valid_z], z_res[valid_z]))
                min_c = sub_pts.min(axis=0)
                voxels = np.floor((sub_pts - min_c) / 1.0).astype(int)
                grid_shape = voxels.max(axis=0) + 1
                if np.prod(grid_shape) < 5_000_000:
                    vox_grid = np.zeros(tuple(grid_shape), dtype=bool)
                    vox_grid[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = True
                    vox_labels, num_comp = ndimage.label(vox_grid, structure=ndimage.generate_binary_structure(3, 1))
                    pt_vox = vox_labels[voxels[:, 0], voxels[:, 1], voxels[:, 2]]
                    vox_sizes = np.bincount(pt_vox, minlength=num_comp + 1)
                    main_comp = pt_vox == np.argmax(vox_sizes[1:]) + 1
                    valid_z_idx = np.where(valid_z)[0]
                    clean_mask = np.zeros(len(face_points), dtype=bool)
                    clean_mask[valid_z_idx[main_comp]] = True
                else:
                    clean_mask = valid_z
            else:
                clean_mask = np.ones(len(face_points), dtype=bool)
            residual_outliers = face_points[~clean_mask]
            face_points = face_points[clean_mask]
        else:
            residual_outliers = np.zeros((0, 3), dtype=np.float64)

        if len(face_points) < decomp_config.min_patch_points:
            raise RuntimeError("Planar surface has too few connected, non-edge points")
        excluded = np.concatenate((
            points[~connected],
            points[connected_idx[edge_mask]],
            stat_outliers,
            residual_outliers,
        ), axis=0)
        roughness = analyze_points(face_points, config=config)
        bbox = np.ptp(face_points, axis=0)
        dims = (float(np.sort(bbox)[2]), float(np.sort(bbox)[1]))
        area = dims[0] * dims[1]
        single_patch = SurfacePatchResult(
            patch_id=1,
            name="Primary Surface",
            normal=tuple(float(x) for x in roughness.plane.normal),
            centroid=tuple(float(x) for x in roughness.plane.centroid),
            area_mm2=area,
            dims_mm=dims,
            point_count=len(face_points),
            is_astm_compliant=bool(area >= decomp_config.astm_min_area_mm2 and min(dims) >= 50.0),
            roughness=roughness,
            points=face_points,
            invalid_points=excluded,
        )
        return ObjectRoughnessResult(
            patches=[single_patch],
            total_points=total_pts,
            assigned_points=len(face_points),
            unassigned_points=len(excluded),
            coverage_pct=100.0 * len(face_points) / total_pts,
            mean_svr_um=roughness.svr_um,
            worst_svr_um=roughness.svr_um,
            best_svr_um=roughness.svr_um,
            dominant_patch=single_patch,
            config=config,
            decomp_config=decomp_config,
            unassigned_points_arr=excluded,
        )

    # 1. Subsample for fast plane discovery
    if progress:
        progress("Discovering dominant 3D planes", 0.15)

    stride = max(1, total_pts // decomp_config.subsample_target)
    sub = points[::stride]

    rng = np.random.default_rng(42)
    rem = sub.copy()
    plane_models: list[tuple[np.ndarray, float, np.ndarray]] = []

    min_inliers = max(
        min(100, len(sub) // (decomp_config.max_faces * 2)),
        decomp_config.min_patch_points // stride,
        int(0.015 * len(sub)),
    )

    for _ in range(decomp_config.max_faces * 2):
        if len(rem) < min_inliers:
            break
        best_inliers: np.ndarray = np.array([], dtype=np.int64)
        best_n: np.ndarray | None = None
        best_d: float | None = None

        triplets = rng.integers(0, len(rem), size=(decomp_config.ransac_iterations, 3))
        for t in triplets:
            p1, p2, p3 = rem[t]
            n = np.cross(p2 - p1, p3 - p1)
            norm = float(np.linalg.norm(n))
            if norm < 1e-6:
                continue
            n /= norm
            d = -float(np.dot(n, p1))
            dists = np.abs(np.dot(rem, n) + d)
            inliers = np.where(dists < decomp_config.plane_distance_thresh_mm)[0]
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_n, best_d = n, d
                if len(inliers) > 0.40 * len(rem):
                    break

        if len(best_inliers) < min_inliers:
            break

        # Refine on inliers with SVD
        in_pts = rem[best_inliers]
        c = np.mean(in_pts, axis=0)
        _, _, v = np.linalg.svd(in_pts - c, full_matrices=False)
        n = v[2] / np.linalg.norm(v[2])
        # Orient normal vector outward from global object centroid
        c_global = np.mean(points, axis=0)
        if np.dot(n, c - c_global) < 0:
            n = -n
        d = -float(np.dot(n, c))

        # Check for duplicate / parallel planes within min_plane_separation_mm
        is_duplicate = False
        cos_parallel = np.cos(np.radians(decomp_config.parallel_angle_thresh_deg))
        for existing_n, existing_d, _ in plane_models:
            dot = abs(float(np.dot(n, existing_n)))
            if dot > cos_parallel:
                sep = abs(float(np.dot(c, existing_n) + existing_d))
                if sep < decomp_config.min_plane_separation_mm:
                    is_duplicate = True
                    break

        # Remove inliers from discovery set
        dists = np.abs(np.dot(rem, n) + d)
        rem = rem[dists >= decomp_config.plane_distance_thresh_mm]

        if is_duplicate:
            continue

        plane_models.append((n, d, c))

        if len(plane_models) >= decomp_config.max_faces:
            break

    if not plane_models:
        raise RuntimeError("No dominant planar faces could be extracted from 3D object scan")

    # 2. Simultaneous Closest-Plane Assignment
    if progress:
        progress("Assigning full-resolution points to faces", 0.35)

    dists_matrix = np.column_stack([np.abs(np.dot(points, n) + d) for n, d, c in plane_models])
    min_dist_plane = np.argmin(dists_matrix, axis=1)
    min_dist_val = np.min(dists_matrix, axis=1)

    edge_tree = cKDTree(points)
    raw_patches: list[dict[str, Any]] = []

    for model_idx, (n, d, c) in enumerate(plane_models):
        cand_mask = (min_dist_plane == model_idx) & (min_dist_val < decomp_config.plane_distance_thresh_mm)
        cand_idx = np.where(cand_mask)[0]
        if len(cand_idx) < decomp_config.min_patch_points:
            continue

        cand_pts = points[cand_idx]

        # ── Phase 1: Spatial component filter ──
        # Remove remote coplanar fixtures / disconnected point islands
        connected = _largest_spatial_component(cand_pts, decomp_config.cluster_cell_mm)
        connected_idx = cand_idx[connected]

        # ── Phase 2: Boundary edge peeling & curvature detection ──
        # Peeling near adjacent non-parallel planes prevents face points from wrapping around corners/chamfers
        connected_pts = points[connected_idx]
        adj_edge_mask = np.zeros(len(connected_pts), dtype=bool)
        if len(plane_models) > 1 and decomp_config.edge_margin_mm > 0:
            cos_thresh = np.cos(np.radians(decomp_config.parallel_angle_thresh_deg))
            for j, (other_n, other_d, _) in enumerate(plane_models):
                if model_idx == j:
                    continue
                dot = abs(float(np.dot(n, other_n)))
                if dot < cos_thresh:
                    signed_dist = np.dot(connected_pts, other_n) + other_d
                    adj_edge_mask |= (signed_dist > -decomp_config.edge_margin_mm)

        # Curvature edge detection along boundaries (for chamfers / freeform transitions)
        curv_edge_mask = _curvature_edge_mask(points, connected_idx, decomp_config, edge_tree)
        edge_mask = adj_edge_mask | curv_edge_mask
        clean_idx = connected_idx[~edge_mask]
        clean_pts = points[clean_idx]

        # ── Phase 3: Statistical outlier removal ──
        # Catch remaining isolated junk with anomalous point spacing
        # (matches C++ StatOutlierRemoval setMeanK=6, setStddevMulThresh=3.0)
        if len(clean_pts) > 10:
            outlier_mask = _stat_outlier_filter(clean_pts, k_neighbors=6, std_mul=3.0)
            face_idx = clean_idx[~outlier_mask]
            face_pts = points[face_idx]
            stat_outliers = clean_pts[outlier_mask]
        else:
            face_idx = clean_idx
            face_pts = clean_pts
            stat_outliers = np.zeros((0, 3), dtype=np.float64)

        # ── Phase 4: Robust out-of-plane residual & 3D connectivity filter ──
        # Reject floating scan artifacts, optical triangulation reflections,
        # and disjointed fragments hovering above/below the true physical surface
        if len(face_pts) > 10:
            c_tmp = np.mean(face_pts, axis=0)
            diff_tmp = face_pts - c_tmp
            _, _, v_tmp = np.linalg.svd((diff_tmp.T @ diff_tmp) / len(face_pts))
            z_res = np.dot(diff_tmp, v_tmp[2])

            med_z = np.median(z_res)
            mad_z = np.median(np.abs(z_res - med_z))
            z_limit = max(0.40, min(decomp_config.plane_distance_thresh_mm, 4.0 * 1.4826 * mad_z))
            valid_z = np.abs(z_res - med_z) <= z_limit

            if np.any(valid_z):
                u_res = np.dot(diff_tmp, v_tmp[0])
                v_res = np.dot(diff_tmp, v_tmp[1])
                sub_pts = np.column_stack((u_res[valid_z], v_res[valid_z], z_res[valid_z]))
                min_c = sub_pts.min(axis=0)
                voxels = np.floor((sub_pts - min_c) / 1.0).astype(int)
                grid_shape = voxels.max(axis=0) + 1
                if np.prod(grid_shape) < 5_000_000:
                    vox_grid = np.zeros(tuple(grid_shape), dtype=bool)
                    vox_grid[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = True
                    vox_labels, num_comp = ndimage.label(vox_grid, structure=ndimage.generate_binary_structure(3, 1))
                    pt_vox = vox_labels[voxels[:, 0], voxels[:, 1], voxels[:, 2]]
                    vox_sizes = np.bincount(pt_vox, minlength=num_comp + 1)
                    main_comp = pt_vox == np.argmax(vox_sizes[1:]) + 1
                    valid_z_idx = np.where(valid_z)[0]
                    clean_mask = np.zeros(len(face_pts), dtype=bool)
                    clean_mask[valid_z_idx[main_comp]] = True
                else:
                    clean_mask = valid_z
            else:
                clean_mask = np.ones(len(face_pts), dtype=bool)

            residual_outliers = face_pts[~clean_mask]
            face_idx = face_idx[clean_mask]
            face_pts = face_pts[clean_mask]
        else:
            residual_outliers = np.zeros((0, 3), dtype=np.float64)

        invalid_pts = np.concatenate((
            cand_pts[~connected],
            points[connected_idx[edge_mask]],
            stat_outliers,
            residual_outliers,
        ), axis=0)
        if len(face_pts) < decomp_config.min_patch_points:
            continue

        # Coordinate projection onto local face plane
        c_face = np.mean(face_pts, axis=0)
        diff = face_pts - c_face
        cov = (diff.T @ diff) / max(len(face_pts), 1)
        _, _, v = np.linalg.svd(cov)
        u_raw, v_raw, normal = v[0], v[1], v[2]
        c_global = np.mean(points, axis=0)
        if np.dot(normal, c_face - c_global) < 0:
            normal = -normal

        # Find minimum area bounding box dimensions in 2D
        u_proj = np.dot(diff, u_raw)
        v_proj = np.dot(diff, v_raw)
        angles = np.linspace(0, np.pi / 2, 45)
        best_area = float("inf")
        best_dims = (float(np.ptp(u_proj)), float(np.ptp(v_proj)))
        for theta in angles:
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rot_u = cos_t * u_proj - sin_t * v_proj
            rot_v = sin_t * u_proj + cos_t * v_proj
            w_u = float(np.ptp(rot_u))
            w_v = float(np.ptp(rot_v))
            if w_u * w_v < best_area:
                best_area = w_u * w_v
                best_dims = (max(w_u, w_v), min(w_u, w_v))

        area_est = best_area
        u_range, v_range = best_dims

        if area_est < decomp_config.min_patch_area_mm2:
            continue

        raw_patches.append(
            {
                "points": face_pts,
                "indices": face_idx,
                "normal": tuple(float(x) for x in normal),
                "centroid": tuple(float(x) for x in c_face),
                "area_mm2": area_est,
                "dims_mm": (round(u_range, 1), round(v_range, 1)),
                "point_count": len(face_pts),
                "invalid_points": invalid_pts,
            }
        )

    if not raw_patches:
        raise RuntimeError("Segmented faces did not meet minimum area requirements")

    # Sort patches deterministically by canonical 3D orientation (+Z, -Z, +X, -X, +Y, -Y)
    def canonical_patch_sort_key(p: dict[str, Any]) -> tuple[int, int, float]:
        n = np.asarray(p["normal"])
        axis = int(np.argmax(np.abs(n)))  # 0 for X, 1 for Y, 2 for Z
        axis_rank = 0 if axis == 2 else (1 if axis == 0 else 2)  # Z (top/bottom) -> X (right/left) -> Y (front/back)
        sign = 0 if n[axis] >= 0 else 1
        return (axis_rank, sign, -float(p["area_mm2"]))

    raw_patches.sort(key=canonical_patch_sort_key)

    # 3. Metrology execution per surface patch
    from .analyze import analyze_points

    patches: list[SurfacePatchResult] = []
    final_assigned_mask = np.zeros(total_pts, dtype=bool)
    total_assigned = 0

    for idx, rp in enumerate(raw_patches):
        patch_id = idx + 1
        pct = 0.50 + 0.45 * (idx / len(raw_patches))
        if progress:
            progress(f"Analyzing ASTM roughness for Face {patch_id}", pct)

        face_pts = rp["points"]
        try:
            r = analyze_points(face_pts, config=config)
        except (ValueError, RuntimeError):
            continue

        final_assigned_mask[rp["indices"]] = True
        total_assigned += len(face_pts)

        # Name based on area rank and dimensions
        dims = rp["dims_mm"]
        is_compliant = bool(rp["area_mm2"] >= decomp_config.astm_min_area_mm2 and min(dims) >= 50.0)
        face_name = f"Face {len(patches) + 1} ({dims[0]:.0f}×{dims[1]:.0f} mm)"

        patch_res = SurfacePatchResult(
            patch_id=len(patches) + 1,
            name=face_name,
            normal=rp["normal"],
            centroid=rp["centroid"],
            area_mm2=rp["area_mm2"],
            dims_mm=dims,
            point_count=len(face_pts),
            is_astm_compliant=is_compliant,
            roughness=r,
            points=face_pts,
            invalid_points=rp.get("invalid_points", np.zeros((0, 3), dtype=np.float64)),
        )
        patches.append(patch_res)

    if not patches:
        raise RuntimeError("Segmented faces could not be processed for ASTM roughness")

    if progress:
        progress("Complete", 1.0)

    # Global summary calculations
    areas = np.array([p.area_mm2 for p in patches])
    svrs = np.array([p.roughness.svr_um for p in patches])
    mean_svr = float(np.sum(areas * svrs) / np.sum(areas))
    worst_svr = float(np.max(svrs))
    best_svr = float(np.min(svrs))
    dominant = patches[0]
    coverage = float(total_assigned / total_pts * 100.0)
    unassigned = total_pts - total_assigned
    unassigned_pts = points[~final_assigned_mask]

    return ObjectRoughnessResult(
        patches=patches,
        total_points=total_pts,
        assigned_points=total_assigned,
        unassigned_points=unassigned,
        coverage_pct=coverage,
        mean_svr_um=mean_svr,
        worst_svr_um=worst_svr,
        best_svr_um=best_svr,
        dominant_patch=dominant,
        config=config,
        decomp_config=decomp_config,
        unassigned_points_arr=unassigned_pts,
    )
