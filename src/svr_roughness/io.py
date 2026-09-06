from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import numpy as np

from .result import FloatArray


def load_points(path: str | Path) -> FloatArray:
    """Load supported scan output into an Nx3 XYZ point array in source units.

    The roughness engine expects millimeters. Most scanner PLY/PCD/STL outputs in
    this repo are already millimeters; callers should scale before analysis if a
    supplier uses another unit convention.
    """

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".ply":
        return load_ply(source)
    if suffix == ".pcd":
        return load_pcd(source)
    if suffix == ".stl":
        return load_stl_vertices(source)
    if suffix == ".obj":
        return load_obj(source)
    if suffix in {".csv", ".xyz", ".txt", ".tsv"}:
        return load_delimited(source)
    if suffix == ".npy":
        return _clean_loaded_points(np.load(source, allow_pickle=False))
    if suffix == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            if not archive.files:
                raise ValueError("NPZ file does not contain an array")
            return _clean_loaded_points(archive[archive.files[0]])
    raise ValueError(f"Unsupported point source type: {suffix}")


def load_ply(path: str | Path) -> FloatArray:
    with Path(path).open("rb") as handle:
        first = handle.readline().strip()
        if first != b"ply":
            raise ValueError("Not a PLY file")

        vertex_count = None
        vertex_properties: list[str] = []
        vertex_types: list[str] = []
        in_vertex_element = False
        data_format = ""
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("PLY header ended before end_header")
            stripped = line.decode("ascii", errors="replace").strip()
            if stripped.startswith("format "):
                parts = stripped.split()
                if len(parts) >= 2:
                    data_format = parts[1]
            if stripped.startswith("element "):
                parts = stripped.split()
                in_vertex_element = parts[1] == "vertex"
                if in_vertex_element:
                    vertex_count = int(parts[2])
                    vertex_properties = []
                    vertex_types = []
            elif in_vertex_element and stripped.startswith("property "):
                parts = stripped.split()
                if len(parts) == 3 and parts[1] != "list":
                    vertex_types.append(parts[1])
                    vertex_properties.append(parts[2])
            elif stripped == "end_header":
                break

        if vertex_count is None:
            raise ValueError("PLY vertex count not found")
        xyz_indexes = _xyz_field_indexes(vertex_properties)
        if data_format == "ascii":
            rows = np.loadtxt(handle, max_rows=vertex_count, usecols=xyz_indexes, dtype=np.float64)
        elif data_format in ("binary_little_endian", "binary_big_endian"):
            byte_order = "<" if data_format == "binary_little_endian" else ">"
            dtype = np.dtype(
                [(name, _ply_numpy_dtype(type_name)) for name, type_name in zip(vertex_properties, vertex_types)]
            ).newbyteorder(byte_order)
            vertices = np.fromfile(handle, dtype=dtype, count=vertex_count)
            rows = np.column_stack([vertices[vertex_properties[index]] for index in xyz_indexes])
        else:
            raise ValueError(f"Unsupported PLY format: {data_format!r}")

    if rows.ndim == 1:
        rows = rows.reshape(1, 3)
    return _clean_loaded_points(rows)


load_ascii_ply = load_ply


def load_delimited(path: str | Path) -> FloatArray:
    suffix = Path(path).suffix.lower()
    delimiter = "," if suffix == ".csv" else "\t" if suffix == ".tsv" else None
    try:
        data = np.loadtxt(path, delimiter=delimiter, comments="#", ndmin=2)
    except ValueError as exc:
        raise ValueError(f"Could not read XYZ coordinates from {Path(path).name}") from exc
    if data.shape[1] < 3:
        raise ValueError("Delimited point files must contain at least three columns")
    return _clean_loaded_points(data[:, :3])


def load_obj(path: str | Path) -> FloatArray:
    vertices = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if parts and parts[0] == "v" and len(parts) >= 4:
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not vertices:
        raise ValueError("No OBJ vertices were found")
    return _clean_loaded_points(np.asarray(vertices, dtype=np.float64))


def _ply_numpy_dtype(type_name: str) -> str:
    dtype = {
        "char": "<i1",
        "int8": "<i1",
        "uchar": "<u1",
        "uint8": "<u1",
        "short": "<i2",
        "int16": "<i2",
        "ushort": "<u2",
        "uint16": "<u2",
        "int": "<i4",
        "int32": "<i4",
        "uint": "<u4",
        "uint32": "<u4",
        "float": "<f4",
        "double": "<f8",
    }.get(type_name)
    if dtype is None:
        raise ValueError(f"Unsupported PLY property type: {type_name}")
    return dtype


def load_pcd(path: str | Path) -> FloatArray:
    with Path(path).open("rb") as handle:
        header, data_start = _read_pcd_header(handle)
        fields = header["FIELDS"].split()
        points_count = int(header["POINTS"])
        data_mode = header.get("DATA", "").lower()

        handle.seek(data_start)
        if data_mode == "ascii":
            points = _read_ascii_pcd(handle, fields, points_count)
        elif data_mode == "binary":
            points = _read_binary_pcd(handle, header, fields, points_count)
        else:
            raise ValueError(f"Unsupported PCD DATA mode: {header.get('DATA')}")
    return _clean_loaded_points(points)


def load_stl_vertices(path: str | Path) -> FloatArray:
    raw = Path(path).read_bytes()
    if _looks_like_binary_stl(raw):
        vertices = _read_binary_stl_vertices(raw)
    else:
        vertices = _read_ascii_stl_vertices(raw.decode("utf-8", errors="ignore"))
    return _clean_loaded_points(np.unique(np.round(vertices, decimals=9), axis=0))


def _read_pcd_header(handle: BinaryIO) -> tuple[dict[str, str], int]:
    header: dict[str, str] = {}
    while True:
        line = handle.readline()
        if not line:
            raise ValueError("PCD header ended before DATA")
        decoded = line.decode("utf-8", errors="replace").strip()
        if not decoded or decoded.startswith("#"):
            continue
        key, _, value = decoded.partition(" ")
        header[key.upper()] = value.strip()
        if key.upper() == "DATA":
            return header, handle.tell()


def _read_ascii_pcd(handle: BinaryIO, fields: list[str], points_count: int) -> FloatArray:
    xyz_indexes = _xyz_field_indexes(fields)
    text = handle.read().decode("utf-8", errors="replace")
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        values = line.split()
        rows.append([float(values[index]) for index in xyz_indexes])
        if len(rows) >= points_count:
            break
    return np.asarray(rows, dtype=np.float64)


def _read_binary_pcd(
    handle: BinaryIO,
    header: dict[str, str],
    fields: list[str],
    points_count: int,
) -> FloatArray:
    sizes = [int(value) for value in header["SIZE"].split()]
    types = header["TYPE"].split()
    counts = [int(value) for value in header.get("COUNT", " ".join(["1"] * len(fields))).split()]
    dtype_fields = []

    for field, size, type_code, count in zip(fields, sizes, types, counts):
        dtype = _numpy_dtype(size, type_code)
        shape = (count,) if count > 1 else ()
        dtype_fields.append((field, dtype, shape))

    dtype = np.dtype(dtype_fields)
    data = np.frombuffer(handle.read(dtype.itemsize * points_count), dtype=dtype, count=points_count)
    return np.column_stack([data[axis].astype(np.float64).reshape(points_count) for axis in ("x", "y", "z")])


def _xyz_field_indexes(fields: list[str]) -> list[int]:
    lowered = [field.lower() for field in fields]
    try:
        return [lowered.index(axis) for axis in ("x", "y", "z")]
    except ValueError as exc:
        raise ValueError("The file must contain x, y, and z fields") from exc


def _numpy_dtype(size: int, type_code: str) -> str:
    if type_code == "F":
        return {4: "<f4", 8: "<f8"}[size]
    if type_code == "I":
        return {1: "<i1", 2: "<i2", 4: "<i4", 8: "<i8"}[size]
    if type_code == "U":
        return {1: "<u1", 2: "<u2", 4: "<u4", 8: "<u8"}[size]
    raise ValueError(f"Unsupported PCD field type: {type_code}")


def _looks_like_binary_stl(raw: bytes) -> bool:
    if len(raw) < 84:
        return False
    triangle_count = int.from_bytes(raw[80:84], byteorder="little", signed=False)
    return 84 + triangle_count * 50 == len(raw)


def _read_binary_stl_vertices(raw: bytes) -> FloatArray:
    triangle_count = int.from_bytes(raw[80:84], byteorder="little", signed=False)
    dtype = np.dtype(
        [
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attribute_byte_count", "<u2"),
        ]
    )
    triangles = np.frombuffer(raw, dtype=dtype, count=triangle_count, offset=84)
    return triangles["vertices"].reshape(-1, 3).astype(np.float64)


def _read_ascii_stl_vertices(text: str) -> FloatArray:
    vertices = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("vertex "):
            _, x, y, z = stripped.split()
            vertices.append((float(x), float(y), float(z)))
    if not vertices:
        raise ValueError("No STL vertices were found")
    return np.asarray(vertices, dtype=np.float64)


def _clean_loaded_points(points: FloatArray) -> FloatArray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected an Nx3 point array")
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        raise ValueError("Need at least 3 valid points")
    return points
