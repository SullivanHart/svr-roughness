"""ctypes bridge to the native SurfInspect numerical core.

There is intentionally no Python numerical fallback in this module. The
published analysis API must use the CGAL/PCL implementation extracted from
Cloud-Viewer.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import NamedTuple

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
        ("grid_width", ctypes.c_size_t),
        ("grid_height", ctypes.c_size_t),
        ("grid_origin_x_mm", ctypes.c_double),
        ("grid_origin_y_mm", ctypes.c_double),
        ("grid_z_mm", ctypes.POINTER(ctypes.c_double)),
        ("grid_svr_um", ctypes.POINTER(ctypes.c_double)),
    ]


class NativeAnalysisResult(NamedTuple):
    """Structured return from the native C++ analysis core."""

    sa_um: float
    sq_um: float
    svr_um: float
    input_points: int
    processed_points: int
    variogram_bins_um: np.ndarray
    variogram_counts: np.ndarray
    surface_distances_mm: np.ndarray
    grid_z_mm: np.ndarray | None
    grid_svr_um: np.ndarray | None
    grid_origin_mm: np.ndarray | None


_LOADED_LIBRARY: ctypes.CDLL | None = None


def _library_candidates() -> list[Path]:
    override = os.environ.get("SVR_ROUGHNESS_NATIVE")
    root = Path(__file__).resolve().parent / "native"
    source_root = Path(__file__).resolve().parents[2]
    build_dll = source_root / "native" / "build" / "Release" / "SurfInspectNative.dll"
    packaged_dll = root / "SurfInspectNative.dll"

    candidates: list[Path] = []
    # Always prioritize the freshest compiled binary between build/Release and packaged copy
    if build_dll.is_file() and packaged_dll.is_file():
        if build_dll.stat().st_mtime >= packaged_dll.stat().st_mtime:
            candidates.extend([build_dll, packaged_dll])
        else:
            candidates.extend([packaged_dll, build_dll])
    elif build_dll.is_file():
        candidates.append(build_dll)
    elif packaged_dll.is_file():
        candidates.append(packaged_dll)

    candidates.extend(
        [
            source_root / "native" / "build" / "Release" / "SurfInspectNative.dll",
            root / "SurfInspectNative.dll",
            source_root / "native" / "build" / "libSurfInspectNative.so",
            source_root / "native" / "build" / "libSurfInspectNative.dylib",
            root / "libSurfInspectNative.so",
            root / "libSurfInspectNative.dylib",
        ]
    )
    if override:
        candidates.insert(0, Path(override))
    return candidates


def _load_library() -> ctypes.CDLL:
    global _LOADED_LIBRARY
    if _LOADED_LIBRARY is not None:
        return _LOADED_LIBRARY

    errors: list[str] = []
    for candidate in _library_candidates():
        if candidate.is_file():
            try:
                if os.name == "nt":
                    # Python 3.8+ does not search the extension directory for
                    # dependent DLLs unless it is explicitly registered.
                    search_dirs = [candidate.parent]
                    # Walk up ancestors to find vcpkg_installed/x64-windows/bin
                    curr = candidate.parent
                    for _ in range(5):
                        vcpkg_bin = curr / "vcpkg_installed" / "x64-windows" / "bin"
                        if vcpkg_bin.is_dir():
                            search_dirs.append(vcpkg_bin)
                            break
                        if curr == curr.parent:
                            break
                        curr = curr.parent
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
                _LOADED_LIBRARY = library
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
) -> NativeAnalysisResult:
    """Run the native C++ SurfInspect analysis core.

    Returns a NativeAnalysisResult with metrics, variogram, distances,
    and (when gaussian_mesh is enabled) the 2D filtered grid produced
    by the C++ ISO 16610-61 Gaussian pipeline.
    """
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
        variogram_bins = np.ctypeslib.as_array(
            native_result.variogram_um, shape=(native_result.variogram_points,)
        ).copy()
        variogram_counts = np.ctypeslib.as_array(
            native_result.variogram_counts, shape=(native_result.variogram_points,)
        ).copy()
        distances = np.ctypeslib.as_array(
            native_result.signed_distances_mm, shape=(native_result.distance_count,)
        ).copy()

        # Extract the 2D grid if the C++ core produced an organized point cloud.
        # This is the case when gaussian_mesh=True (the standard-compliant path).
        grid_z: np.ndarray | None = None
        grid_svr: np.ndarray | None = None
        grid_origin: np.ndarray | None = None
        if native_result.grid_width > 1 and native_result.grid_height > 1 and native_result.grid_z_mm:
            total = native_result.grid_width * native_result.grid_height
            grid_flat = np.ctypeslib.as_array(
                native_result.grid_z_mm, shape=(total,)
            ).copy()
            # PCL organized clouds store points row-major: height rows x width cols.
            grid_z = grid_flat.reshape((native_result.grid_height, native_result.grid_width))
            if native_result.grid_svr_um:
                grid_svr_flat = np.ctypeslib.as_array(
                    native_result.grid_svr_um, shape=(total,)
                ).copy()
                grid_svr = grid_svr_flat.reshape((native_result.grid_height, native_result.grid_width))
            grid_origin = np.array([
                native_result.grid_origin_x_mm,
                native_result.grid_origin_y_mm,
                config.grid_mm,
            ], dtype=np.float64)

        return NativeAnalysisResult(
            sa_um=float(native_result.sa_um),
            sq_um=float(native_result.sq_um),
            svr_um=float(native_result.svr_um),
            input_points=int(native_result.input_points),
            processed_points=int(native_result.processed_points),
            variogram_bins_um=variogram_bins,
            variogram_counts=variogram_counts,
            surface_distances_mm=distances,
            grid_z_mm=grid_z,
            grid_svr_um=grid_svr,
            grid_origin_mm=grid_origin,
        )
    finally:
        library.si_free_result(ctypes.byref(native_result))
