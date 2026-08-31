"""ctypes bridge to the native SurfInspect numerical core.

There is intentionally no Python numerical fallback in this module. The
published analysis API must use the CGAL/PCL implementation extracted from
Cloud-Viewer.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .config import RoughnessConfig

_DLL_DIRECTORY_HANDLES: list[object] = []


class _NativeConfig(ctypes.Structure):
    _fields_ = [
        ("voxel_size_m", ctypes.c_double),
        ("mesh_grid_size_m", ctypes.c_double),
        ("target_edge_length_m", ctypes.c_double),
        ("variogram_points", ctypes.c_int),
        ("variogram_span_m", ctypes.c_double),
        ("long_cutoff_m", ctypes.c_double),
        ("short_cutoff_m", ctypes.c_double),
        ("statistical_filter", ctypes.c_int),
        ("statistical_mean_k", ctypes.c_int),
        ("statistical_stddev", ctypes.c_double),
        ("gaussian_mesh", ctypes.c_int),
    ]


class _NativeResult(ctypes.Structure):
    _fields_ = [
        ("sa_um", ctypes.c_double),
        ("sq_um", ctypes.c_double),
        ("svr_um", ctypes.c_double),
        ("input_points", ctypes.c_size_t),
        ("processed_points", ctypes.c_size_t),
        ("variogram_points", ctypes.c_size_t),
        ("variogram_um", ctypes.POINTER(ctypes.c_double)),
        ("variogram_counts", ctypes.POINTER(ctypes.c_size_t)),
        ("distance_count", ctypes.c_size_t),
        ("signed_distances_mm", ctypes.POINTER(ctypes.c_double)),
    ]


def _library_candidates() -> list[Path]:
    override = os.environ.get("SVR_ROUGHNESS_NATIVE")
    root = Path(__file__).resolve().parent / "native"
    candidates = [
        root / "SurfInspectNative.dll",
        root / "libSurfInspectNative.so",
        root / "libSurfInspectNative.dylib",
    ]
    # Editable installs keep the compiled library in the source checkout.
    source_root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            source_root / "native" / "build" / "Release" / "SurfInspectNative.dll",
            source_root / "native" / "build" / "libSurfInspectNative.so",
            source_root / "native" / "build" / "libSurfInspectNative.dylib",
        ]
    )
    if override:
        candidates.insert(0, Path(override))
    return candidates


def _load_library() -> ctypes.CDLL:
    errors: list[str] = []
    for candidate in _library_candidates():
        if candidate.is_file():
            try:
                if os.name == "nt":
                    # Python 3.8+ does not search the extension directory for
                    # dependent DLLs unless it is explicitly registered.
                    search_dirs = [candidate.parent]
                    package_bin = candidate.parent.parent.parent / "vcpkg_installed" / "x64-windows" / "bin"
                    if package_bin.is_dir():
                        search_dirs.append(package_bin)
                    for directory in search_dirs:
                        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(directory)))
                library = ctypes.WinDLL(str(candidate)) if os.name == "nt" else ctypes.CDLL(str(candidate))
                library.si_analyze_points.argtypes = [
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.c_size_t,
                    ctypes.POINTER(_NativeConfig),
                    ctypes.POINTER(_NativeResult),
                ]
                library.si_analyze_points.restype = ctypes.c_int
                library.si_free_result.argtypes = [ctypes.POINTER(_NativeResult)]
                library.si_free_result.restype = None
                library.si_last_error.argtypes = []
                library.si_last_error.restype = ctypes.c_char_p
                return library
            except OSError as exc:
                errors.append(f"{candidate}: {exc}")
                continue
        else:
            errors.append(f"{candidate}: file does not exist")
    locations = ", ".join(str(path) for path in _library_candidates())
    raise RuntimeError(
        "The native SurfInspect core is not available in this installation. "
        "Build the wheel with CGAL/PCL available, or set "
        f"SVR_ROUGHNESS_NATIVE to a compatible shared library. Checked: {locations}. "
        f"Load errors: {'; '.join(errors)}"
    )


def analyze_native(
    points_xyz_mm: npt.ArrayLike, config: RoughnessConfig
) -> tuple[float, float, float, int, int, np.ndarray, np.ndarray, np.ndarray]:
    points = np.ascontiguousarray(np.asarray(points_xyz_mm, dtype=np.float64))
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected an Nx3 point array")
    if len(points) < 20:
        raise ValueError("SurfInspect native core requires at least 20 points")

    native_config = _NativeConfig(
        voxel_size_m=config.grid_mm * 0.001,
        mesh_grid_size_m=config.mesh_resolution_mm * 0.001,
        target_edge_length_m=0.005,
        variogram_points=config.svr_points,
        variogram_span_m=config.svr_span_mm * 0.001,
        long_cutoff_m=config.long_cutoff_mm * 0.001,
        short_cutoff_m=config.short_cutoff_mm * 0.001,
        statistical_filter=int(config.statistical_filter),
        statistical_mean_k=config.statistical_mean_k,
        statistical_stddev=config.statistical_stddev,
        gaussian_mesh=int(config.gaussian_mesh),
    )
    native_result = _NativeResult()
    library = _load_library()
    status = library.si_analyze_points(
        points.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(points),
        ctypes.byref(native_config),
        ctypes.byref(native_result),
    )
    if status:
        message = library.si_last_error().decode("utf-8", errors="replace")
        raise RuntimeError(f"Native SurfInspect analysis failed: {message}")
    try:
        bins = np.ctypeslib.as_array(
            native_result.variogram_um, shape=(native_result.variogram_points,)
        ).copy()
        counts = np.ctypeslib.as_array(
            native_result.variogram_counts, shape=(native_result.variogram_points,)
        ).copy()
        distances = np.ctypeslib.as_array(
            native_result.signed_distances_mm, shape=(native_result.distance_count,)
        ).copy()
        return (
            float(native_result.sa_um),
            float(native_result.sq_um),
            float(native_result.svr_um),
            int(native_result.input_points),
            int(native_result.processed_points),
            bins,
            counts,
            distances,
        )
    finally:
        library.si_free_result(ctypes.byref(native_result))
