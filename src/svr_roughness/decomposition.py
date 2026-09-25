from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .algorithm import analyze_pure_python
from .config import RoughnessConfig
from .io import load_points
from .result import RoughnessResult, format_report


@dataclass(frozen=True)
class DecompositionConfig:
    """Configuration for 3D multi-patch surface decomposition."""

    plane_distance_thresh_mm: float = 2.0
    edge_margin_mm: float = 1.5
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

        roughness = analyze_points(points, config=config)
        bbox = np.ptp(points, axis=0)
        dims = (float(np.sort(bbox)[2]), float(np.sort(bbox)[1]))
        area = dims[0] * dims[1]
        single_patch = SurfacePatchResult(
            patch_id=1,
            name="Primary Surface",
            normal=tuple(float(x) for x in roughness.plane.normal),
            centroid=tuple(float(x) for x in roughness.plane.centroid),
            area_mm2=area,
            dims_mm=dims,
            point_count=total_pts,
            is_astm_compliant=bool(area >= decomp_config.astm_min_area_mm2 and min(dims) >= 50.0),
            roughness=roughness,
            points=points,
        )
        return ObjectRoughnessResult(
            patches=[single_patch],
            total_points=total_pts,
            assigned_points=total_pts,
            unassigned_points=0,
            coverage_pct=100.0,
            mean_svr_um=roughness.svr_um,
            worst_svr_um=roughness.svr_um,
            best_svr_um=roughness.svr_um,
            dominant_patch=single_patch,
            config=config,
            decomp_config=decomp_config,
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

    # Estimate nominal object dimensions from detected opposite parallel planes (if any)
    opp_dists: list[float] = []
    for i in range(len(plane_models)):
        for j in range(i + 1, len(plane_models)):
            ni, di, ci = plane_models[i]
            nj, dj, cj = plane_models[j]
            if float(np.dot(ni, nj)) < -0.8:
                opp_dists.append(abs(float(np.dot(ci, nj) + dj)))
    nom_size = float(np.median(opp_dists)) if opp_dists else None

    # 2. Assign full-resolution points to planes
    if progress:
        progress("Assigning full-resolution points to faces", 0.35)

    assigned = np.zeros(total_pts, dtype=bool)
    raw_patches: list[dict[str, Any]] = []

    for model_idx, (n, d, c) in enumerate(plane_models):
        dists = np.abs(np.dot(points, n) + d)
        cand_idx = np.where((dists < decomp_config.plane_distance_thresh_mm) & (~assigned))[0]
        if len(cand_idx) < decomp_config.min_patch_points:
            continue

        cand_pts = points[cand_idx]

        # Edge removal: peel away points within edge_margin_mm of non-parallel adjacent planes
        # Points outside or within margin of adjacent plane have signed_dist > -edge_margin_mm
        if decomp_config.edge_margin_mm > 0 and len(plane_models) > 1:
            edge_mask = np.zeros(len(cand_pts), dtype=bool)
            for j, (other_n, other_d, _) in enumerate(plane_models):
                if model_idx == j:
                    continue
                # Check if non-parallel (angle > threshold)
                dot = abs(float(np.dot(n, other_n)))
                cos_thresh = np.cos(np.radians(decomp_config.parallel_angle_thresh_deg))
                if dot < cos_thresh:
                    signed_dist = np.dot(cand_pts, other_n) + other_d
                    # Signed distance: > 0 means outside adjacent boundary; [-edge_margin_mm, 0] is edge margin
                    edge_mask |= (signed_dist > -decomp_config.edge_margin_mm)
                    # If this adjacent direction has no opposite plane, bound by nominal thickness if known
                    has_opp = any(
                        float(np.dot(other_n, ok_n)) < -0.8
                        for k, (ok_n, _, _) in enumerate(plane_models)
                        if k != j
                    )
                    if not has_opp and nom_size is not None:
                        edge_mask |= (signed_dist < -(nom_size - decomp_config.edge_margin_mm))

            pure_pts = cand_pts[~edge_mask]
            face_idx = cand_idx[~edge_mask]
            if len(pure_pts) < decomp_config.min_patch_points:
                continue
            face_pts = pure_pts
        else:
            face_pts = cand_pts
            face_idx = cand_idx

        # Coordinate projection onto local face plane
        c_face = np.mean(face_pts, axis=0)
        diff = face_pts - c_face
        cov = (diff.T @ diff) / len(face_pts)
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

        assigned[cand_idx] = True

        raw_patches.append(
            {
                "points": face_pts,
                "indices": face_idx,
                "normal": tuple(float(x) for x in normal),
                "centroid": tuple(float(x) for x in c_face),
                "area_mm2": area_est,
                "dims_mm": (round(u_range, 1), round(v_range, 1)),
                "point_count": len(face_pts),
            }
        )

    if not raw_patches:
        raise RuntimeError("Segmented faces did not meet minimum area requirements")

    # Sort patches by area descending
    raw_patches.sort(key=lambda p: p["area_mm2"], reverse=True)

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
