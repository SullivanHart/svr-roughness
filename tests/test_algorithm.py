import unittest
import numpy as np
from svr_roughness._algorithm import voxel_downsample, sa_sq


class TestAlgorithm(unittest.TestCase):
    def test_voxel_downsample(self):
        points = np.array([
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1],
            [1.0, 1.0, 1.0],
        ])
        downsampled = voxel_downsample(points, leaf_mm=0.5)
        self.assertEqual(len(downsampled), 2)
        np.testing.assert_allclose(downsampled[0], [0.05, 0.05, 0.05])

    def test_sa_sq(self):
        grid = np.array([
            [0.001, -0.001],
            [0.002, -0.002],
        ])
        valid = np.ones_like(grid, dtype=bool)
        sa, sq = sa_sq(grid, valid)
        self.assertEqual(sa, 1.5)  # mean(|z|*1000)
        np.testing.assert_allclose(sq, np.sqrt(2.5))


if __name__ == "__main__":
    unittest.main()
