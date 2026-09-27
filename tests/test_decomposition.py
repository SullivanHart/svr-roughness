from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scipy.spatial import cKDTree
from svr_roughness.decomposition import _curvature_edge_mask, _largest_spatial_component

from svr_roughness import (
    DecompositionConfig,
    ObjectRoughnessResult,
    RoughnessConfig,
    analyze_file,
    analyze_object,
    decompose_3d_object,
    is_planar_surface,
)


def _generate_synthetic_plane(
    origin: tuple[float, float, float],
    u_vec: tuple[float, float, float],
    v_vec: tuple[float, float, float],
    length_u: float = 60.0,
    length_v: float = 60.0,
    step: float = 0.6,
    noise_sigma: float = 0.02,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(42)
    u_vals = np.arange(0, length_u, step)
    v_vals = np.arange(0, length_v, step)
    uu, vv = np.meshgrid(u_vals, v_vals)
    uu = uu.ravel()
    vv = vv.ravel()

    u_dir = np.array(u_vec, dtype=float)
    u_dir /= np.linalg.norm(u_dir)
    v_dir = np.array(v_vec, dtype=float)
    v_dir /= np.linalg.norm(v_dir)
    n_dir = np.cross(u_dir, v_dir)
    n_dir /= np.linalg.norm(n_dir)

    pts = (
        np.array(origin, dtype=float)
        + uu[:, None] * u_dir[None, :]
        + vv[:, None] * v_dir[None, :]
        + rng.normal(0, noise_sigma, size=len(uu))[:, None] * n_dir[None, :]
    )
    return pts


def test_is_planar_surface_single_plane() -> None:
    rng = np.random.default_rng(123)
    plane_pts = _generate_synthetic_plane(
        (0, 0, 0), (1, 0, 0), (0, 1, 0), length_u=60.0, length_v=60.0, step=1.0, noise_sigma=0.01, rng=rng
    )
    assert is_planar_surface(plane_pts) is True


def test_is_planar_surface_3d_box() -> None:
    rng = np.random.default_rng(123)
    p1 = _generate_synthetic_plane((0, 0, 0), (1, 0, 0), (0, 1, 0), length_u=60.0, length_v=60.0, step=1.0, rng=rng)
    p2 = _generate_synthetic_plane((0, 0, 0), (1, 0, 0), (0, 0, 1), length_u=60.0, length_v=60.0, step=1.0, rng=rng)
    p3 = _generate_synthetic_plane((0, 0, 0), (0, 1, 0), (0, 0, 1), length_u=60.0, length_v=60.0, step=1.0, rng=rng)
    box_pts = np.vstack([p1, p2, p3])
    assert is_planar_surface(box_pts) is False


def test_synthetic_box_decomposition() -> None:
    rng = np.random.default_rng(42)
    # Generate 3 orthogonal faces of a cube (60x60 mm each)
    f1 = _generate_synthetic_plane((0, 0, 0), (1, 0, 0), (0, 1, 0), length_u=60.0, length_v=60.0, step=0.6, rng=rng)
    f2 = _generate_synthetic_plane((0, 0, 0), (1, 0, 0), (0, 0, 1), length_u=60.0, length_v=60.0, step=0.6, rng=rng)
    f3 = _generate_synthetic_plane((0, 0, 0), (0, 1, 0), (0, 0, 1), length_u=60.0, length_v=60.0, step=0.6, rng=rng)
    box_pts = np.vstack([f1, f2, f3])

    decomp_cfg = DecompositionConfig(
        plane_distance_thresh_mm=1.0,
        edge_margin_mm=2.0,
        min_patch_points=2000,
        min_patch_area_mm2=400.0,
        astm_min_area_mm2=2500.0,
        max_faces=6,
    )
    rough_cfg = RoughnessConfig(grid_mm=0.5, svr_points=25)

    result = decompose_3d_object(box_pts, config=rough_cfg, decomp_config=decomp_cfg)

    assert isinstance(result, ObjectRoughnessResult)
    assert result.patch_count == 3
    assert result.total_points == len(box_pts)
    assert result.assigned_points > 0
    assert result.coverage_pct > 80.0
    assert result.mean_svr_um > 0.0
    assert result.worst_svr_um >= result.best_svr_um

    for p in result.patches:
        assert p.area_mm2 >= 2000.0
        assert p.roughness.svr_um > 0.0
        # ASTM WK92969 §3.1.5: ~60x60mm minus 2mm edge margins is ~56x56mm > 50x50mm
        assert p.is_astm_compliant is True

    # Test serialization and reporting
    d = result.to_dict()
    assert "patches" in d
    assert len(d["patches"]) == 3
    assert d["patch_count"] == 3

    report = result.format_report()
    assert "Multi-Surface 3D Object Roughness Report" in report
    assert "Face 1" in report
    assert "Face 2" in report
    assert "Face 3" in report


def test_single_planar_surface_passthrough() -> None:
    rng = np.random.default_rng(99)
    plane_pts = _generate_synthetic_plane(
        (0, 0, 0), (1, 0, 0), (0, 1, 0), length_u=60.0, length_v=60.0, step=0.8, rng=rng
    )
    rough_cfg = RoughnessConfig(grid_mm=0.5, svr_points=25)
    result = decompose_3d_object(plane_pts, config=rough_cfg)

    assert result.patch_count == 1
    assert result.coverage_pct >= 90.0  # stat outlier filter may clip a few boundary pts
    assert result.dominant_patch.is_astm_compliant is True
    assert result.dominant_patch.roughness.svr_um > 0.0


def test_analyze_file_auto_decompose(tmp_path: Path) -> None:
    rng = np.random.default_rng(77)
    f1 = _generate_synthetic_plane((0, 0, 0), (1, 0, 0), (0, 1, 0), length_u=55.0, length_v=55.0, step=0.8, rng=rng)
    f2 = _generate_synthetic_plane((0, 0, 0), (0, 0, 1), (0, 1, 0), length_u=55.0, length_v=55.0, step=0.8, rng=rng)
    pts = np.vstack([f1, f2])

    test_csv = tmp_path / "test_corner.csv"
    np.savetxt(test_csv, pts, delimiter=",", header="x,y,z", comments="")

    decomp_cfg = DecompositionConfig(min_patch_points=1000, edge_margin_mm=1.0)
    rough_cfg = RoughnessConfig(grid_mm=0.5, svr_points=25)

    res = analyze_file(test_csv, config=rough_cfg, auto_decompose=True, decomp_config=decomp_cfg)
    assert isinstance(res, ObjectRoughnessResult)
    assert res.patch_count == 2


@pytest.mark.skipif(
    not Path("scans/SCRATA Cubes/2021.02.22_SCRATA_CUBE_Mix.pcd").exists(),
    reason="Scan data not present in environment",
)
def test_real_scrata_cube_decomposition() -> None:
    pcd_path = Path("scans/SCRATA Cubes/2021.02.22_SCRATA_CUBE_Mix.pcd")
    cfg = RoughnessConfig(grid_mm=0.5, svr_points=25)
    decomp_cfg = DecompositionConfig(
        plane_distance_thresh_mm=1.5,
        edge_margin_mm=3.0,
        subsample_target=80000,
        ransac_iterations=200,
    )

    result = analyze_object(pcd_path, config=cfg, decomp_config=decomp_cfg)
    assert result.patch_count >= 4  # The cube scan has 5 dominant faces
    assert result.total_points > 700000
    assert result.worst_svr_um > result.best_svr_um
    assert result.worst_svr_um > 100.0  # Mixed cube contains a very rough face (>A4)


def test_curvature_detects_corner_without_relying_on_other_plane_models() -> None:
    coords = np.arange(0.0, 20.0, 0.5)
    x, y = np.meshgrid(coords, coords)
    top = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    side = np.column_stack((x.ravel(), np.zeros(x.size), y.ravel()))
    cloud = np.vstack((top, side))
    cfg = DecompositionConfig(edge_curvature_threshold=0.02)
    top_indices = np.arange(len(top))
    edges = _curvature_edge_mask(cloud, top_indices, cfg, cKDTree(cloud))
    near_corner = top[:, 1] < 1.0
    interior = top[:, 1] > 8.0
    assert edges[near_corner].mean() > 0.5
    assert edges[interior].mean() < 0.05


def test_interior_rough_spot_is_not_removed_as_an_edge() -> None:
    coords = np.arange(0.0, 30.0, 0.5)
    x, y = np.meshgrid(coords, coords)
    z = 2.0 * np.exp(-((x - 15.0) ** 2 + (y - 15.0) ** 2) / 2.0)
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    mask = _curvature_edge_mask(points, np.arange(len(points)), DecompositionConfig(), cKDTree(points))
    central_spot = (x.ravel() - 15.0) ** 2 + (y.ravel() - 15.0) ** 2 < 4.0
    assert not mask[central_spot].any()


def test_spatial_component_excludes_remote_coplanar_fixture() -> None:
    x, y = np.meshgrid(np.arange(0.0, 20.0), np.arange(0.0, 20.0))
    surface = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    fixture = surface[:16].copy() + np.array([100.0, 0.0, 0.0])
    keep = _largest_spatial_component(np.vstack((surface, fixture)), 2.0)
    assert keep[:len(surface)].all()
    assert not keep[len(surface):].any()
