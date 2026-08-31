from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

from ._display_grid import (
    apply_filters,
    crop_points,
    fill_holes_neighbor_mean,
    fit_plane_basis,
    grid_residuals,
)
from .config import RoughnessConfig, coerce_range
from .io import load_ascii_ply, load_points
from .native import analyze_native
from .result import PlaneFit, RoughnessGrid, RoughnessResult


def analyze_points(
    points_xyz_mm: npt.ArrayLike,
    config: RoughnessConfig | None = None,
    progress: Callable[[str, float], None] | None = None,
    **overrides: Any,
) -> RoughnessResult:
    """Analyze an Nx3 XYZ point array using the native SurfInspect C++ core.

    This is the canonical roughness path. File-specific APIs load data and
    then delegate here to execute the native C++ numerical engine.
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
        raise ValueError("SurfInspect native core requires at least 20 valid points")

    # Strictly execute the native C++ core extracted from Cloud-Viewer
    native = analyze_native(selected, resolved)
    sa_um = native[0]
    sq_um = native[1]
    svr_um = native[2]
    cropped_points = native[4]
    variogram_bins_um = native[5]
    variogram_counts = native[6]
    surface_distances_mm = native[7]

    # Compute 2D grid matrix for visualization heatmaps
    plane = fit_plane_basis(selected)
    cropped, _ = crop_points(plane.coords, resolved)
    grid_raw, valid_raw, grid_origin = grid_residuals(cropped, resolved.grid_mm, resolved.min_points_per_cell)
    grid_filled, valid_filled = fill_holes_neighbor_mean(grid_raw, valid_raw, resolved.max_hole_passes)
    grid_filtered = apply_filters(grid_filled, valid_filled, resolved.grid_mm, resolved.short_cutoff_mm, resolved.long_cutoff_mm)
    grid_filtered = grid_filtered - np.nanmean(grid_filtered[valid_filled])

    if progress:
        progress("Complete", 1.0)
    return RoughnessResult(
        sa_um=sa_um,
        sq_um=sq_um,
        svr_um=svr_um,
        points=int(len(points)),
        cropped_points=cropped_points,
        plane=plane,
        raw_residual_std_mm=float(plane.coords[:, 2].std()),
        raw_residual_p05_mm=float(np.percentile(plane.coords[:, 2], 5)),
        raw_residual_p95_mm=float(np.percentile(plane.coords[:, 2], 95)),
        grid=RoughnessGrid(grid_raw, grid_filled, grid_filtered, valid_raw, valid_filled, grid_origin),
        surface_distances_mm=surface_distances_mm,
        variogram_bins_um=variogram_bins_um,
        variogram_counts=variogram_counts,
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