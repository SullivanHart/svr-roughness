import os
import unittest
from pathlib import Path

import numpy as np

from svr_roughness.algorithm import (
    analyze_pure_python,
    apply_dual_pass_gaussian_filter,
    compute_heatmap_grid,
    compute_variogram_and_svr,
    detect_surface_abnormalities,
    interpolate_heatmap_at_points,
    iso_gaussian_filter_1d_kernel,
    pca_align_plane,
    rasterize_elevation_grid,
    voxel_downsample,
)
from svr_roughness.analyze import analyze_points
from svr_roughness.config import RoughnessConfig
from svr_roughness.result import format_report
from svr_roughness.io import load_points


class TestAlgorithm(unittest.TestCase):
    def test_voxel_downsample(self):
        pts = np.array([
            [0.0001, 0.0001, 0.0001],
            [0.00015, 0.00015, 0.00015],
            [0.01, 0.01, 0.01],
        ])
        down = voxel_downsample(pts, voxel_size_m=0.001)
        self.assertEqual(len(down), 2)
        np.testing.assert_allclose(down[0], [0.000125, 0.000125, 0.000125])

    def test_voxel_downsample_empty_and_single(self):
        empty = np.empty((0, 3), dtype=np.float64)
        down_empty = voxel_downsample(empty, 0.001)
        self.assertEqual(len(down_empty), 0)

        single = np.array([[1.0, 2.0, 3.0]])
        down_single = voxel_downsample(single, 0.001)
        self.assertEqual(len(down_single), 1)
        np.testing.assert_allclose(down_single[0], [1.0, 2.0, 3.0])

    def test_pca_align_plane(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(-1.0, 1.0, 500)
        y = rng.uniform(-1.0, 1.0, 500)
        z = 0.5 * x - 0.3 * y + 2.0
        pts = np.column_stack([x, y, z])
        rotated, centroid, eig_vecs = pca_align_plane(pts)

        # Normalization and right-handedness
        np.testing.assert_allclose(eig_vecs.T @ eig_vecs, np.eye(3), atol=1e-10)
        self.assertGreater(np.linalg.det(eig_vecs), 0.0)

        # Planar residuals near zero
        self.assertAlmostEqual(float(np.mean(rotated[:, 2])), 0.0, places=6)
        self.assertLess(float(np.std(rotated[:, 2])), 1e-6)

    def test_rasterize_elevation_grid(self):
        # Create a 2D regular grid of points
        x, y = np.meshgrid(np.linspace(0, 0.01, 20), np.linspace(0, 0.01, 20))
        z = np.sin(x * 100) * 0.0001
        pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

        grid_z, valid, ox, oy = rasterize_elevation_grid(pts, pitch_m=0.001)
        self.assertGreater(grid_z.shape[0], 5)
        self.assertGreater(grid_z.shape[1], 5)
        self.assertTrue(np.all(valid))
        self.assertFalse(np.any(np.isnan(grid_z)))

    def test_gaussian_filter_iso16610(self):
        k = iso_gaussian_filter_1d_kernel(cutoff_m=0.001, pitch_m=0.0002)
        # Kernel must sum to 1.0 (energy preservation)
        self.assertAlmostEqual(float(np.sum(k)), 1.0, places=7)
        # Kernel must be symmetric
        np.testing.assert_allclose(k, k[::-1], atol=1e-10)

        # Constant grid should filter to zero roughness after long-pass subtraction
        flat = np.ones((50, 50)) * 0.005
        rough = apply_dual_pass_gaussian_filter(flat, pitch_m=0.0002, short_cutoff_m=0.0004, long_cutoff_m=0.002)
        np.testing.assert_allclose(rough, 0.0, atol=1e-8)

    def test_variogram_monotonicity_on_roughness(self):
        rng = np.random.default_rng(123)
        grid_z = rng.normal(0, 0.00005, (100, 100))
        sa, sq, svr, var_bins, var_counts = compute_variogram_and_svr(
            grid_z, pitch_m=0.0002, points_on_var=10, span_m=0.0005
        )
        self.assertGreater(sa, 0.0)
        self.assertGreater(sq, 0.0)
        self.assertGreater(svr, 0.0)
        # Cauchy-Schwarz invariant: Sq >= Sa
        self.assertGreaterEqual(sq, sa)
        self.assertEqual(len(var_bins), 10)
        self.assertTrue(np.all(var_counts > 0))

        grid_svr = compute_heatmap_grid(grid_z * 1000.0, pitch_mm=0.2, radius_mm=2.0)
        self.assertEqual(grid_svr.shape, (100, 100))
        self.assertTrue(np.any(np.isfinite(grid_svr)))

    def test_variogram_uses_only_measured_pairs_and_all_configured_bins(self):
        grid = np.full((8, 8), 0.0001)
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 2:6] = True
        sa, sq, svr, bins, counts = compute_variogram_and_svr(
            grid, pitch_m=0.001, points_on_var=5, span_m=0.001, valid_mask=mask
        )
        self.assertAlmostEqual(sa, 100.0)
        self.assertAlmostEqual(sq, 100.0)
        self.assertAlmostEqual(svr, 0.0, delta=1e-4)
        self.assertEqual(counts[0], 0)
        self.assertLess(sum(counts), 8 * 8 * 5 * 5)
        self.assertTrue(np.all(np.isfinite(bins)))

        # Only the first occupied distance bucket contributes to the ASTM divisor.
        step = np.array([[0.0, 0.0001]])
        _, _, small_svr, small_bins, small_counts = compute_variogram_and_svr(
            step, pitch_m=0.001, points_on_var=4, span_m=0.002
        )
        self.assertEqual(int(np.count_nonzero(small_counts)), 1)
        self.assertAlmostEqual(small_svr, float(small_bins.sum() / 4))

        masked_step = np.full((4, 4), np.nan)
        masked_step[1, 1] = 0.0
        masked_step[1, 2] = 0.0001
        _, _, masked_svr, masked_bins, masked_counts = compute_variogram_and_svr(
            masked_step, pitch_m=0.001, points_on_var=2, span_m=0.002
        )
        self.assertEqual(masked_counts.tolist(), [2, 0])
        self.assertAlmostEqual(masked_bins[0], 100.0 / np.sqrt(2), places=5)
        self.assertAlmostEqual(masked_svr, 100.0 / (2 * np.sqrt(2)), places=5)

    def test_analyze_pure_python_synthetic(self):
        # 1000 points with Gaussian noise
        rng = np.random.default_rng(99)
        x = rng.uniform(-5.0, 5.0, 2000)
        y = rng.uniform(-5.0, 5.0, 2000)
        z = rng.normal(0.0, 0.05, 2000)
        pts = np.column_stack([x, y, z])

        res = analyze_pure_python(pts, voxel_size_mm=0.2, short_cutoff_mm=0.5, long_cutoff_mm=5.0)
        self.assertGreater(res.processed_points, 100)
        self.assertGreater(res.sa_um, 0.0)
        self.assertGreater(res.sq_um, res.sa_um * 0.99)
        self.assertGreater(res.svr_um, 0.0)
        self.assertEqual(len(res.variogram_bins_um), 10)

    def test_pure_python_on_scrata(self):
        sample_pcd = Path(__file__).resolve().parents[2] / "SurfaceRoughnessHeatmap" / "samples" / "SCRATA_A1.pcd"
        if not sample_pcd.is_file():
            self.skipTest(f"Test file {sample_pcd} not found")

        points = load_points(sample_pcd)
        res = analyze_pure_python(points)
        self.assertGreater(res.processed_points, 100000)
        self.assertAlmostEqual(res.svr_um, 28.35, delta=0.5)
        self.assertAlmostEqual(res.sa_um, 33.12, delta=0.5)
        self.assertAlmostEqual(res.sq_um, 42.48, delta=0.5)
        self.assertEqual(len(res.variogram_bins_um), 10)

    def test_compute_heatmap_grid_disc_averaging_toggle(self):
        # Create a 2D surface grid with an isolated sharp peak in the center
        grid = np.zeros((40, 40), dtype=np.float64)
        grid[20, 20] = 0.05  # 50 µm peak

        # 1. Point-centered mode (default / disc_averaging=False) preserves the sharp peak
        hmap_point = compute_heatmap_grid(grid, pitch_mm=0.2, radius_mm=2.0, disc_averaging=False)
        self.assertTrue(np.all(np.isnan(hmap_point) | (hmap_point >= 0.0)))
        peak_point = hmap_point[20, 20]

        # 2. Disc-averaging mode (disc_averaging=True) dilutes the peak across the whole disc
        hmap_disc = compute_heatmap_grid(grid, pitch_mm=0.2, radius_mm=2.0, disc_averaging=True)
        peak_disc = hmap_disc[20, 20]

        # Point-centered response at the peak itself should be significantly sharper/higher
        self.assertGreater(peak_point, peak_disc * 2.0)

        # 3. Test environment variable toggle gate
        old_env = os.environ.get("SVR_DISC_AVERAGING")
        try:
            os.environ["SVR_DISC_AVERAGING"] = "1"
            hmap_env_disc = compute_heatmap_grid(grid, pitch_mm=0.2, radius_mm=2.0)
            np.testing.assert_allclose(hmap_env_disc, hmap_disc)

            os.environ["SVR_DISC_AVERAGING"] = "0"
            hmap_env_point = compute_heatmap_grid(grid, pitch_mm=0.2, radius_mm=2.0)
            np.testing.assert_allclose(hmap_env_point, hmap_point)
        finally:
            if old_env is None:
                os.environ.pop("SVR_DISC_AVERAGING", None)
            else:
                os.environ["SVR_DISC_AVERAGING"] = old_env

    def test_interpolate_heatmap_at_points(self):
        # 4x4 grid with known values
        hmap = np.array([
            [10.0, 20.0, 30.0, 40.0],
            [10.0, 20.0, 30.0, 40.0],
            [10.0, 20.0, 30.0, 40.0],
            [10.0, 20.0, 30.0, 40.0],
        ], dtype=np.float64)

        # Points exactly on grid vertices (pitch = 1.0, origin = 0, 0)
        u = np.array([0.0, 1.0, 0.5])
        v = np.array([0.0, 0.0, 0.0])
        interp = interpolate_heatmap_at_points(hmap, 0.0, 0.0, 1.0, u, v)
        self.assertAlmostEqual(interp[0], 10.0)
        self.assertAlmostEqual(interp[1], 20.0)
        self.assertAlmostEqual(interp[2], 15.0)  # midpoint between 10 and 20

        # Empty points array
        empty_interp = interpolate_heatmap_at_points(hmap, 0.0, 0.0, 1.0, np.array([]), np.array([]))
        self.assertEqual(len(empty_interp), 0)

        # Degenerate grid (< 2 in any dimension) returns NaNs
        deg_grid = np.array([[10.0, 20.0]])
        deg_interp = interpolate_heatmap_at_points(deg_grid, 0.0, 0.0, 1.0, np.array([0.0]), np.array([0.0]))
        self.assertTrue(np.isnan(deg_interp[0]))

    def test_detect_surface_abnormalities_none(self):
        # Clean smooth grid has no abnormalities
        grid = np.zeros((50, 50), dtype=np.float64)
        abn, mask = detect_surface_abnormalities(grid, pitch_mm=0.2, global_svr_um=10.0)
        self.assertEqual(len(abn), 0)
        self.assertFalse(np.any(mask))

    def test_detect_surface_abnormalities_protrusion_and_depression(self):
        # 100x100 grid at 0.2 mm pitch = 20x20 mm
        n = 100
        pitch = 0.2
        grid = np.zeros((n, n), dtype=np.float64)

        # Base roughness: small noise (std ~ 0.002 mm = 2 µm)
        rng = np.random.default_rng(123)
        grid += rng.normal(0.0, 0.002, (n, n))

        # 1. Protrusion: scab of radius 1.5 mm (d = 3.0 mm, area ~ 7.1 mm²) at (20, 20)
        # Height +0.050 mm = +50 µm
        y, x = np.ogrid[:n, :n]
        dist_prot = np.sqrt(((x - 20) * pitch) ** 2 + ((y - 20) * pitch) ** 2)
        grid[dist_prot <= 1.5] += 0.050

        # 2. Depression: pit of radius 1.5 mm at (80, 80)
        # Depth -0.050 mm = -50 µm
        dist_dep = np.sqrt(((x - 80) * pitch) ** 2 + ((y - 80) * pitch) ** 2)
        grid[dist_dep <= 1.5] -= 0.050

        global_svr = 5.0  # µm
        abn, mask = detect_surface_abnormalities(
            grid,
            pitch_mm=pitch,
            global_svr_um=global_svr,
            threshold_factor=2.0,
            min_area_mm2=2.0,
        )

        self.assertEqual(len(abn), 2)
        self.assertTrue(np.any(mask))

        kinds = {a.kind for a in abn}
        self.assertEqual(kinds, {"protrusion", "depression"})

        for a in abn:
            self.assertGreaterEqual(a.area_mm2, 2.0)
            self.assertGreaterEqual(a.peak_deviation_um, 40.0)

    def test_analyze_points_abnormality_rerun(self):
        # 60x60 grid of 3D points
        pitch = 0.2
        xs = np.linspace(0, 12.0, 60)
        ys = np.linspace(0, 12.0, 60)
        xx, yy = np.meshgrid(xs, ys)
        zz = np.zeros_like(xx)

        # Add defect (> 2x local variance and area > 2 mm²)
        dist_scab = np.sqrt((xx - 6.0) ** 2 + (yy - 6.0) ** 2)
        zz[dist_scab <= 1.5] += 0.080  # 80 µm height deviation

        pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        conf = RoughnessConfig(
            grid_mm=0.2,
            detect_abnormalities=True,
            min_abnormality_area_mm2=2.0,
        )
        res = analyze_points(pts, config=conf)

        self.assertGreaterEqual(len(res.abnormalities), 1)
        self.assertIsNotNone(res.svr_clean_um)
        # Clean S_VR excluding the defect should be strictly lower than uncorrected S_VR
        self.assertLess(res.svr_clean_um, res.svr_um)
        self.assertGreater(res.abnormal_area_pct, 0.0)
        self.assertGreater(res.abnormal_frequency_per_dm2, 0.0)

        # Check ASTM report contains abnormality section with correct mm and µm units
        report = format_report(res)
        self.assertIn("Abnormality Analysis (ASTM WK92969 §8.7 / §10.1.5.3)", report)
        self.assertIn(f"Clean S_VR (w/o abn):      {(res.svr_clean_um / 1000.0):.4f} mm ({res.svr_clean_um:.1f} µm)", report)
        self.assertIn(f"#{res.abnormalities[0].id}", report)


if __name__ == "__main__":
    unittest.main()
