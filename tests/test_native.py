import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from svr_roughness import RoughnessConfig, analyze_file, analyze_points, format_report


class TestRoughnessAnalysis(unittest.TestCase):
    def test_synthetic_points_analysis(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(-10.0, 10.0, 1000)
        y = rng.uniform(-10.0, 10.0, 1000)
        z = 0.05 * x - 0.02 * y + rng.normal(0, 0.02, 1000)
        points = np.column_stack([x, y, z])

        config = RoughnessConfig(gaussian_mesh=True, grid_mm=0.5)
        result = analyze_points(points, config=config)

        self.assertGreater(result.points, 0)
        self.assertGreater(result.sa_um, 0.0)
        self.assertGreater(result.sq_um, 0.0)
        self.assertGreater(result.svr_um, 0.0)
        self.assertIsNotNone(result.grid)

        # Test report formatting
        report = format_report(result)
        self.assertIn("ASTM WK92969 Surface Variogram Roughness", report)
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
        self.assertGreater(result.cropped_points, 100000)
        self.assertTrue(25.0 < result.sa_um < 45.0)
        self.assertTrue(35.0 < result.sq_um < 55.0)
        self.assertTrue(20.0 < result.svr_um < 35.0)


if __name__ == "__main__":
    unittest.main()
