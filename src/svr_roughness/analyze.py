from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .decomposition import DecompositionConfig, ObjectRoughnessResult

from ._display_grid import crop_points, fit_plane_basis
from .algorithm import analyze_pure_python
from .config import RoughnessConfig, coerce_range
from .io import load_ascii_ply, load_points
from .result import RoughnessResult


def analyze_points(
    points_xyz_mm: npt.ArrayLike,
    config: RoughnessConfig | None = None,
    progress: Callable[[str, float], None] | None = None,
    **overrides: Any,
) -> RoughnessResult:
    """Analyze an Nx3 XYZ point array using ASTM WK92969 Gaussian areal roughness.

    This is the canonical roughness path. File-specific APIs load data and
    then delegate here to execute the ASTM WK92969 / ISO 16610-61 numerical engine.

    All computation (voxel downsampling, PCA plane alignment, 2.5D grid rasterization,
    dual-pass Gaussian filtration, and FFT autocorrelation variogram) is
    performed using vectorized NumPy operations.
    """

    resolved = _resolve_config(config, overrides)
    if progress:
        progress("Preparing point cloud", 0.15)
    points = np.asarray(points_xyz_mm, dtype=np.float64)
    finite = np.isfinite(points).all(axis=1) if points.ndim == 2 else np.array([])
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected an Nx3 point array")
    points = points[finite]
    if len(points) < 3:
        raise ValueError("Requires at least 3 valid points")

    selected = points
    if resolved.x_range:
        selected = selected[(selected[:, 0] >= resolved.x_range[0]) & (selected[:, 0] <= resolved.x_range[1])]
    if resolved.y_range:
        selected = selected[(selected[:, 1] >= resolved.y_range[0]) & (selected[:, 1] <= resolved.y_range[1])]
    if resolved.z_range:
        selected = selected[(selected[:, 2] >= resolved.z_range[0]) & (selected[:, 2] <= resolved.z_range[1])]
    if resolved.percentile_crop > 0:
        selection_plane = fit_plane_basis(selected)
        _, crop_mask = crop_points(selection_plane.coords, resolved)
        selected = selected[crop_mask]
    if len(selected) < 20:
        raise ValueError("SurfInspect engine requires at least 20 valid points")

    if progress:
        progress("Executing ASTM WK92969 analysis", 0.40)

    # ── Execute the pure-Python ASTM WK92969 numerical engine ──
    pure_res = analyze_pure_python(
        selected,
        voxel_size_mm=resolved.grid_mm,
        short_cutoff_mm=resolved.short_cutoff_mm,
        long_cutoff_mm=resolved.long_cutoff_mm,
        variogram_points=resolved.svr_points,
        variogram_span_mm=resolved.svr_span_mm,
        subtract_noise=resolved.subtract_noise,
    )

    # ── Plane fit synchronized with grid rasterization coordinate frame ──
    from .result import PlaneFit
    if pure_res.plane_centroid_mm is not None:
        diff_sel = selected - pure_res.plane_centroid_mm
        coords_sel = np.column_stack((
            np.dot(diff_sel, pure_res.plane_x_axis),
            np.dot(diff_sel, pure_res.plane_y_axis),
            np.dot(diff_sel, pure_res.plane_normal),
        ))
        plane = PlaneFit(
            centroid=pure_res.plane_centroid_mm,
            normal=pure_res.plane_normal,
            x_axis=pure_res.plane_x_axis,
            y_axis=pure_res.plane_y_axis,
            coords=coords_sel,
        )
    else:
        plane = fit_plane_basis(selected)

    if progress:
        progress("Complete", 1.0)

    return RoughnessResult(
        svr_um=pure_res.svr_um,
        sa_um=pure_res.sa_um,
        sq_um=pure_res.sq_um,
        noise_floor_um=pure_res.noise_floor_um,
        svr_raw_um=pure_res.svr_raw_um,
        points=len(points),
        processed_points=pure_res.processed_points,
        plane=plane,
        grid=pure_res.grid_z_mm,
        elevation_grid=pure_res.elevation_grid_mm,
        grid_pitch_mm=float(resolved.grid_mm),
        grid_origin_mm=pure_res.grid_origin_mm,
        variogram_bins_um=pure_res.variogram_bins_um,
        variogram_counts=pure_res.variogram_counts,
        config=resolved,
    )


def analyze_file(
    path: str | Path,
    config: RoughnessConfig | None = None,
    progress: Callable[[str, float], None] | None = None,
    auto_decompose: bool = False,
    decomp_config: DecompositionConfig | None = None,
    **overrides: Any,
) -> RoughnessResult | ObjectRoughnessResult:
    if auto_decompose:
        return analyze_object(
            path,
            config=config,
            decomp_config=decomp_config,
            progress=progress,
            **overrides,
        )
    if progress:
        progress("Loading scan", 0.05)
    points = load_points(path)
    return analyze_points(points, config, progress=progress, **overrides)


def analyze_object(
    points_or_file: str | Path | npt.ArrayLike,
    config: RoughnessConfig | None = None,
    decomp_config: DecompositionConfig | None = None,
    progress: Callable[[str, float], None] | None = None,
    **overrides: Any,
) -> ObjectRoughnessResult:
    """Analyze a 3D object scan by segmenting it into ASTM WK92969 planar faces.

    Parameters
    ----------
    points_or_file : str, Path, or Nx3 array
        Point cloud file path or Nx3 XYZ coordinate array in mm.
    config : RoughnessConfig, optional
        Metrology configuration (grid pitch, cutoffs, noise subtraction).
    decomp_config : DecompositionConfig, optional
        Decomposition parameters (plane distance, edge margins, min area).
    progress : callable, optional
        Progress callback taking (status_text, percent_0_to_1).
    **overrides : Any
        Keyword arguments to override fields in RoughnessConfig.

    Returns
    -------
    ObjectRoughnessResult
        Container holding segmented surface patches and multi-face metrology.
    """
    from .decomposition import decompose_3d_object

    resolved = _resolve_config(config, overrides)
    return decompose_3d_object(
        points_or_file,
        config=resolved,
        decomp_config=decomp_config,
        progress=progress,
    )


def analyze_ply(
    path: str | Path,
    config: RoughnessConfig | None = None,
    progress: Callable[[str, float], None] | None = None,
    **overrides: Any,
) -> RoughnessResult:
    return analyze_points(load_ascii_ply(path), config, progress=progress, **overrides)


# Backward-compatible names from the first extraction.
compute_roughness = analyze_points
compute_roughness_from_ply = analyze_ply


def _resolve_config(config: RoughnessConfig | None, overrides: dict[str, Any]) -> RoughnessConfig:
    base = config or RoughnessConfig()
    if not overrides:
        return base

    allowed = {field.name for field in fields(RoughnessConfig)}
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        names = ", ".join(unknown)
        raise TypeError(f"Unknown roughness config override(s): {names}")

    normalized = dict(overrides)
    for key in ("x_range", "y_range", "z_range"):
        if key in normalized:
            normalized[key] = coerce_range(normalized[key])
    return replace(base, **normalized)
