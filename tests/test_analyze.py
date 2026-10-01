import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from svr_roughness.result import PlaneFit, RoughnessResult

from svr_roughness import RoughnessConfig, analyze_file, analyze_points, compute_heatmap_grid, format_report


class TestRoughnessAnalysis(unittest.TestCase):
    def test_synthetic_points_analysis(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(-10.0, 10.0, 1000)
        y = rng.uniform(-10.0, 10.0, 1000)
        z = 0.05 * x - 0.02 * y + rng.normal(0, 0.02, 1000)
        points = np.column_stack([x, y, z])

        config = RoughnessConfig(grid_mm=0.5)
        result = analyze_points(points, config=config)

        self.assertGreater(result.points, 0)
        self.assertGreater(result.sa_um, 0.0)
        self.assertGreater(result.sq_um, 0.0)
        self.assertGreater(result.svr_um, 0.0)
        self.assertIsNotNone(result.grid)
        self.assertEqual(result.grid.ndim, 2)

        # Test on-demand heatmap computation
        hmap = result.heatmap(radius_mm=2.5)
        self.assertEqual(hmap.shape, result.grid.shape)

        # Test clean string representation
        repr_str = repr(result)
        self.assertIn("Svr =", repr_str)
        self.assertIn("Sa  =", repr_str)

        # Test report formatting
        report = format_report(result)
        self.assertIn("Surface Variogram Roughness", report)
        self.assertIn("Sa", report)

        # Test metrics JSON save
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            result.save_metrics_json(tmp_path)
            loaded = json.loads(tmp_path.read_text(encoding="utf-8"))
            self.assertIn("sa_um", loaded)
            self.assertIn("sq_um", loaded)
            self.assertIn("svr_um", loaded)
        finally:
            tmp_path.unlink()

    def test_report_without_plane_coordinates(self):
        plane = PlaneFit(
            centroid=np.zeros(3), normal=np.array([0., 0., 1.]),
            x_axis=np.array([1., 0., 0.]), y_axis=np.array([0., 1., 0.]),
            coords=None,
        )
        for grid in (None, np.zeros((3, 5))):
            with self.subTest(has_grid=grid is not None):
                result = RoughnessResult(
                    svr_um=30., sa_um=20., sq_um=25., plane=plane,
                    grid=grid, grid_pitch_mm=0.2, config=RoughnessConfig(),
                )
                self.assertAlmostEqual(result.patch_width_mm, 1. if grid is not None else 0.)
                self.assertAlmostEqual(result.patch_height_mm, 0.6 if grid is not None else 0.)
                self.assertIn("Surface Variogram Roughness", format_report(result))

    def test_too_few_points(self):
        few_points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        with self.assertRaises(ValueError):
            analyze_points(few_points)

    def test_scrata_sample(self):
        sample_pcd = Path(__file__).resolve().parents[2] / "SurfInspect" / "TestFiles" / "SCRATA_A1.pcd"
        if not sample_pcd.is_file():
            self.skipTest(f"Test file {sample_pcd} not found")

        result = analyze_file(sample_pcd)
        self.assertEqual(result.points, 1000255)
        self.assertGreater(result.processed_points, 100000)
        self.assertTrue(25.0 < result.sa_um < 45.0)
        self.assertTrue(35.0 < result.sq_um < 55.0)
        self.assertTrue(20.0 < result.svr_um < 35.0)

    def test_sparse_scan_density_error_and_warning(self):
        # Scan with ~1.0 mm point spacing
        rng = np.random.default_rng(123)
        coords = np.arange(0.0, 50.0, 1.0)
        gx, gy = np.meshgrid(coords, coords)
        pts = np.column_stack([gx.ravel(), gy.ravel(), rng.normal(0, 0.02, size=gx.size)])

        # At default 0.20 mm pitch, spacing ~1.0 mm is > 2.0x pitch -> should raise informative ValueError
        with self.assertRaises(ValueError) as cm:
            analyze_points(pts, config=RoughnessConfig(grid_mm=0.20))
        err_msg = str(cm.exception)
        self.assertIn("Insufficient contiguous surface area at grid_mm=0.20 mm", err_msg)
        self.assertIn("Average point spacing for this scan is ~", err_msg)
        self.assertIn("--grid-mm 0.6", err_msg)

        # Re-running with suggested pitch 0.60 mm succeeds and flags density warning
        res = analyze_points(pts, config=RoughnessConfig(grid_mm=0.60))
        self.assertGreater(res.svr_um, 0.0)
        report = format_report(res)
        self.assertIn("Point Density Status:      WARNING (> 0.20 mm required)", report)

    def test_b1_ply_density_behavior(self):
        b1_path = Path(__file__).resolve().parents[2] / "scans" / "B1.ply"
        if not b1_path.is_file():
            self.skipTest(f"Test scan {b1_path} not found")

        # 1. 0.20 mm grid raises ValueError with suggestion 0.6
        with self.assertRaises(ValueError) as cm:
            analyze_file(b1_path, config=RoughnessConfig(grid_mm=0.20))
        self.assertIn("Insufficient contiguous surface area at grid_mm=0.20 mm", str(cm.exception))
        self.assertIn("--grid-mm 0.6", str(cm.exception))

        # 2. 0.60 mm grid succeeds with warning
        res = analyze_file(b1_path, config=RoughnessConfig(grid_mm=0.60))
        rep = format_report(res)
        self.assertIn("Point Density Status:      WARNING (> 0.20 mm required)", rep)


if __name__ == "__main__":
    unittest.main()

