from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ._display_grid import crop_points, fit_plane_basis
from .algorithm import analyze_pure_python
from .config import RoughnessConfig, coerce_range
from .io import load_ascii_ply, load_points
from .result import RoughnessGrid, RoughnessResult


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
    )

    # ── Plane fit for metadata ──
    plane = fit_plane_basis(selected)

    # ── Package the raster elevation grid into RoughnessGrid ──
    grid_z = pure_res.grid_z_mm
    valid = ~np.isnan(grid_z)
    grid = RoughnessGrid(
        raw=grid_z,
        filled=grid_z,
        filtered=grid_z,
        valid_raw=valid,
        valid_filled=valid,
        origin=pure_res.grid_origin_mm,
        svr_map=pure_res.grid_svr_um,
    )

    if progress:
        progress("Complete", 1.0)
    return RoughnessResult(
        sa_um=pure_res.sa_um,
        sq_um=pure_res.sq_um,
        svr_um=pure_res.svr_um,
        points=len(points),
        cropped_points=pure_res.processed_points,
        plane=plane,
        raw_residual_std_mm=float(plane.coords[:, 2].std()),
        raw_residual_p05_mm=float(np.percentile(plane.coords[:: max(1, len(plane.coords) // 50000), 2], 5)),
        raw_residual_p95_mm=float(np.percentile(plane.coords[:: max(1, len(plane.coords) // 50000), 2], 95)),
        grid=grid,
        surface_distances_mm=pure_res.surface_distances_mm,
        variogram_bins_um=pure_res.variogram_bins_um,
        variogram_counts=pure_res.variogram_counts,
        config=resolved,
    )


def analyze_file(
    path: str | Path,
    config: RoughnessConfig | None = None,
    progress: Callable[[str, float], None] | None = None,
    **overrides: Any,
) -> RoughnessResult:
    if progress:
        progress("Loading scan", 0.05)
    points = load_points(path)
    return analyze_points(points, config, progress=progress, **overrides)


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
