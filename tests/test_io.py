import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from svr_roughness.io import (
    load_ascii_ply,
    load_delimited,
    load_obj,
    load_ply,
    load_points,
    load_stl_vertices,
)


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

    def test_load_binary_ply(self):
        header = (
            b"ply\n"
            b"format binary_little_endian 1.0\n"
            b"element vertex 3\n"
            b"property float x\n"
            b"property float y\n"
            b"property float z\n"
            b"end_header\n"
        )
        data = struct.pack("<3f3f3f", 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5)
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".ply", delete=False) as tmp:
            tmp.write(header + data)
            tmp_path = Path(tmp.name)

        try:
            pts = load_ply(tmp_path)
            self.assertEqual(pts.shape, (3, 3))
            np.testing.assert_allclose(pts[0], [1.5, 2.5, 3.5])
            np.testing.assert_allclose(pts[2], [7.5, 8.5, 9.5])
        finally:
            tmp_path.unlink()

    def test_load_ascii_stl(self):
        content = (
            "solid test\n"
            "  facet normal 0 0 1\n"
            "    outer loop\n"
            "      vertex 0.0 0.0 0.0\n"
            "      vertex 1.0 0.0 0.0\n"
            "      vertex 0.0 1.0 0.0\n"
            "    endloop\n"
            "  endfacet\n"
            "endsolid test\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".stl", delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            pts = load_stl_vertices(tmp_path)
            self.assertEqual(pts.shape, (3, 3))
            np.testing.assert_allclose(pts[0], [0.0, 0.0, 0.0])
            np.testing.assert_allclose(pts[1], [1.0, 0.0, 0.0])
            np.testing.assert_allclose(pts[2], [0.0, 1.0, 0.0])
        finally:
            tmp_path.unlink()

    def test_load_binary_stl(self):
        header = b"\x00" * 80
        num_triangles = struct.pack("<I", 1)
        facet = struct.pack(
            "<3f 3f 3f 3f H",
            0.0, 0.0, 1.0,
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            0,
        )
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".stl", delete=False) as tmp:
            tmp.write(header + num_triangles + facet)
            tmp_path = Path(tmp.name)

        try:
            pts = load_stl_vertices(tmp_path)
            self.assertEqual(pts.shape, (3, 3))
            np.testing.assert_allclose(pts[0], [1.0, 2.0, 3.0])
            np.testing.assert_allclose(pts[2], [7.0, 8.0, 9.0])
        finally:
            tmp_path.unlink()

    def test_load_delimited_csv_and_xyz(self):
        csv_content = "x,y,z\n1.0,2.0,3.0\n4.0,5.0,6.0\n7.0,8.0,9.0\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp.write(csv_content)
            tmp_csv = Path(tmp.name)

        xyz_content = "# XYZ point cloud\n1.0 2.0 3.0\n4.0 5.0 6.0\n7.0 8.0 9.0\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as tmp:
            tmp.write(xyz_content)
            tmp_xyz = Path(tmp.name)

        try:
            pts_csv = load_delimited(tmp_csv)
            pts_xyz = load_delimited(tmp_xyz)
            np.testing.assert_allclose(pts_csv, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            np.testing.assert_allclose(pts_xyz, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        finally:
            tmp_csv.unlink()
            tmp_xyz.unlink()

    def test_load_obj(self):
        obj_content = (
            "# Wavefront OBJ\n"
            "v 1.0 2.0 3.0\n"
            "v 4.0 5.0 6.0\n"
            "v 7.0 8.0 9.0\n"
            "f 1 2 3\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".obj", delete=False) as tmp:
            tmp.write(obj_content)
            tmp_path = Path(tmp.name)

        try:
            pts = load_obj(tmp_path)
            self.assertEqual(pts.shape, (3, 3))
            np.testing.assert_allclose(pts[1], [4.0, 5.0, 6.0])
        finally:
            tmp_path.unlink()

    def test_load_npy_and_npz(self):
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp_npy:
            np.save(tmp_npy, arr)
            npy_path = Path(tmp_npy.name)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp_npz:
            np.savez(tmp_npz, points=arr)
            npz_path = Path(tmp_npz.name)

        try:
            pts_npy = load_points(npy_path)
            pts_npz = load_points(npz_path)
            np.testing.assert_allclose(pts_npy, arr)
            np.testing.assert_allclose(pts_npz, arr)
        finally:
            npy_path.unlink()
            npz_path.unlink()

    def test_unsupported_format(self):
        with tempfile.NamedTemporaryFile(suffix=".unknown", delete=False) as tmp:
            tmp.write(b"random bytes")
            tmp_path = Path(tmp.name)

        try:
            with self.assertRaises(ValueError):
                load_points(tmp_path)
        finally:
            tmp_path.unlink()


if __name__ == "__main__":
    unittest.main()
