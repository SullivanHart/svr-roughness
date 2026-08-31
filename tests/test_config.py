import unittest
from svr_roughness import RoughnessConfig


class TestConfig(unittest.TestCase):
    def test_roughness_config_defaults(self):
        config = RoughnessConfig()
        self.assertEqual(config.grid_mm, 0.30)
        self.assertEqual(config.svr_points, 10)
        self.assertEqual(config.svr_span_mm, 0.5)
        self.assertEqual(config.long_cutoff_mm, 25.0)
        self.assertEqual(config.short_cutoff_mm, 1.0)
        self.assertFalse(config.statistical_filter)
        self.assertTrue(config.gaussian_mesh)

    def test_surfinspect_defaults(self):
        config = RoughnessConfig.surfinspect_defaults()
        self.assertEqual(config.svr_points, 10)
        self.assertEqual(config.long_cutoff_mm, 25.0)
        self.assertEqual(config.short_cutoff_mm, 1.0)


if __name__ == "__main__":
    unittest.main()
