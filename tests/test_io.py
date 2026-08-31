import unittest
import tempfile
from pathlib import Path
import numpy as np
from svr_roughness.io import load_ply, load_ascii_ply


class TestIO(unittest.TestCase):
    def test_load_ascii_ply(self):
        content = (
            "ply\n"
            "format ascii 1.0\n"
            "element vertex 3\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
            "1.0 2.0 3.0 255 0 0\n"
            "4.0 5.0 6.0 0 255 0\n"
            "7.0 8.0 9.0 0 0 255\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ply", delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            pts = load_ascii_ply(tmp_path)
            self.assertEqual(pts.shape, (3, 3))
            np.testing.assert_allclose(pts[0], [1.0, 2.0, 3.0])
            np.testing.assert_allclose(pts[2], [7.0, 8.0, 9.0])
        finally:
            tmp_path.unlink()


if __name__ == "__main__":
    unittest.main()

