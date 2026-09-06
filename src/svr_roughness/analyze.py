from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ._display_grid import crop_points, fit_plane_basis
from .config import RoughnessConfig, coerce_range
from .io import load_ascii_ply, load_points
from .native import analyze_native
from .result import RoughnessGrid, RoughnessResult


def analyze_points(
    points_xyz_mm: npt.ArrayLike,
    config: RoughnessConfig | None = None,
    progress: Callable[[str, float], None] | None = None,
    **overrides: Any,
) -> RoughnessResult:
    """Analyze an Nx3 XYZ point array using the native SurfInspect C++ core.

    This is the canonical roughness path. File-specific APIs load data and
    then delegate here to execute the native C++ numerical engine.

    All heavy computation (gridding, Gaussian filtering, variogram) is
    performed by the C++ core. Python only handles I/O, validation, and
    packaging the result.
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

    # ── Execute the native C++ core (all heavy math happens here) ──
    native = analyze_native(selected, resolved)

    # ── Plane fit for metadata ──
    plane = fit_plane_basis(selected)

    # ── Package the C++ grid into RoughnessGrid ──
    if native.grid_z_mm is not None and native.grid_origin_mm is not None:
        grid_z = native.grid_z_mm
        valid = ~np.isnan(grid_z)
        grid = RoughnessGrid(
            raw=grid_z,
            filled=grid_z,
            filtered=grid_z,
            valid_raw=valid,
            valid_filled=valid,
            origin=native.grid_origin_mm,
            svr_map=native.grid_svr_um,
        )
    else:
        # Non-standard path (gaussian_mesh=False) — no organized grid available.
        # Create a minimal 1×1 placeholder so the result type is always valid.
        empty = np.full((1, 1), np.nan, dtype=np.float64)
        grid = RoughnessGrid(
            raw=empty,
            filled=empty,
            filtered=empty,
            valid_raw=np.zeros((1, 1), dtype=bool),
            valid_filled=np.zeros((1, 1), dtype=bool),
            origin=np.array([0.0, 0.0, resolved.grid_mm], dtype=np.float64),
        )

    if progress:
        progress("Complete", 1.0)
    return RoughnessResult(
        sa_um=native.sa_um,
        sq_um=native.sq_um,
        svr_um=native.svr_um,
        points=len(points),
        cropped_points=native.processed_points,
        plane=plane,
        raw_residual_std_mm=float(plane.coords[:, 2].std()),
        raw_residual_p05_mm=float(np.percentile(plane.coords[:: max(1, len(plane.coords) // 50000), 2], 5)),
        raw_residual_p95_mm=float(np.percentile(plane.coords[:: max(1, len(plane.coords) // 50000), 2], 95)),
        grid=grid,
        surface_distances_mm=native.surface_distances_mm,
        variogram_bins_um=native.variogram_bins_um,
        variogram_counts=native.variogram_counts,
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
