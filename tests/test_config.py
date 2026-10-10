import unittest

from svr_roughness import RoughnessConfig


class TestConfig(unittest.TestCase):
    def test_roughness_config_defaults(self):
        config = RoughnessConfig()
        self.assertEqual(config.grid_mm, 0.20)
        self.assertEqual(config.svr_points, 10)
        self.assertEqual(config.svr_span_mm, 0.5)
        self.assertEqual(config.long_cutoff_mm, 25.0)
        self.assertEqual(config.short_cutoff_mm, 1.0)
        self.assertEqual(config.percentile_crop, 0.0)
        self.assertTrue(config.subtract_noise)
        self.assertTrue(config.gaussian_mesh)

    def test_surfinspect_defaults(self):
        config = RoughnessConfig.surfinspect_defaults()
        self.assertEqual(config.svr_points, 10)
        self.assertEqual(config.long_cutoff_mm, 25.0)
        self.assertEqual(config.short_cutoff_mm, 1.0)

    def test_astm_standard_defaults(self):
        config = RoughnessConfig.astm_standard_defaults()
        self.assertEqual(config.grid_mm, 0.20)
        self.assertEqual(config.svr_points, 10)
        self.assertEqual(config.short_cutoff_mm, 1.0)
        self.assertEqual(config.long_cutoff_mm, 25.0)

    def test_parse_range_and_coerce_range(self):
        from svr_roughness.config import coerce_range, config_from_values, parse_range

        self.assertIsNone(parse_range(None))
        self.assertEqual(parse_range("-5.5:10.2"), (-5.5, 10.2))

        self.assertIsNone(coerce_range(None))
        self.assertEqual(coerce_range("0:50"), (0.0, 50.0))
        self.assertEqual(coerce_range((1.0, 2.0)), (1.0, 2.0))
        self.assertEqual(coerce_range([3, 4]), (3.0, 4.0))

        cfg = config_from_values(
            grid_mm=0.5,
            x_range="-10:10",
            y_range=[0, 20],
            disc_averaging=True,
            gaussian_mesh=False,
        )
        self.assertEqual(cfg.grid_mm, 0.5)
        self.assertEqual(cfg.x_range, (-10.0, 10.0))
        self.assertEqual(cfg.y_range, (0.0, 20.0))
        self.assertIsNone(cfg.z_range)
        self.assertTrue(cfg.disc_averaging)
        self.assertFalse(cfg.gaussian_mesh)


if __name__ == "__main__":
    unittest.main()
