import unittest
from pathlib import Path

import numpy as np

from svr_roughness.algorithm import (
    analyze_pure_python,
    apply_dual_pass_gaussian_filter,
    compute_variogram_and_svr,
    iso_gaussian_filter_1d_kernel,
    pca_align_plane,
    rasterize_elevation_grid,
    voxel_downsample,
)
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
        sa, sq, svr, var_bins, var_counts, grid_svr = compute_variogram_and_svr(
            grid_z, pitch_m=0.0002, points_on_var=10, span_m=0.0005
        )
        self.assertGreater(sa, 0.0)
        self.assertGreater(sq, 0.0)
        self.assertGreater(svr, 0.0)
        # Cauchy-Schwarz invariant: Sq >= Sa
        self.assertGreaterEqual(sq, sa)
        self.assertEqual(len(var_bins), 10)
        self.assertTrue(np.all(var_counts > 0))
        self.assertEqual(grid_svr.shape, (100, 100))

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
        sample_pcd = Path(__file__).resolve().parents[2] / "SurfInspect" / "TestFiles" / "SCRATA_A1.pcd"
        if not sample_pcd.is_file():
            self.skipTest(f"Test file {sample_pcd} not found")

        points = load_points(sample_pcd)
        res = analyze_pure_python(points)
        self.assertGreater(res.processed_points, 100000)
        self.assertAlmostEqual(res.svr_um, 28.51, delta=0.5)
        self.assertAlmostEqual(res.sa_um, 33.74, delta=0.5)
        self.assertAlmostEqual(res.sq_um, 43.43, delta=0.5)
        self.assertEqual(len(res.variogram_bins_um), 10)


if __name__ == "__main__":
    unittest.main()
