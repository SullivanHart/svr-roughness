import unittest
from pathlib import Path
import numpy as np

from svr_roughness import RoughnessConfig, analyze_file, analyze_points


class TestNativeEngine(unittest.TestCase):
    def test_native_synthetic_points(self):
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

    def test_native_too_few_points(self):
        few_points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        with self.assertRaises(ValueError):
            analyze_points(few_points)

    def test_native_scrata_sample(self):
        sample_pcd = Path(__file__).resolve().parents[2] / "SurfInspect" / "TestFiles" / "SCRATA_A1.pcd"
        if not sample_pcd.is_file():
            self.skipTest(f"Test file {sample_pcd} not found")

        result = analyze_file(sample_pcd)
        self.assertEqual(result.points, 1000255)
        self.assertEqual(result.cropped_points, 142380)
        self.assertTrue(30.0 < result.sa_um < 50.0)
        self.assertTrue(40.0 < result.sq_um < 65.0)
        self.assertTrue(20.0 < result.svr_um < 35.0)


if __name__ == "__main__":
    unittest.main()
