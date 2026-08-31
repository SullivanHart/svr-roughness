from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .config import RoughnessConfig

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class PlaneFit:
    centroid: FloatArray
    normal: FloatArray
    x_axis: FloatArray
    y_axis: FloatArray
    coords: FloatArray


@dataclass(frozen=True)
class RoughnessGrid:
    raw: FloatArray
    filled: FloatArray
    filtered: FloatArray
    valid_raw: BoolArray
    valid_filled: BoolArray
    origin: FloatArray


@dataclass(frozen=True)
class RoughnessResult:
    sa_um: float
    sq_um: float
    svr_um: float
    points: int
    cropped_points: int
    plane: PlaneFit
    raw_residual_std_mm: float
    raw_residual_p05_mm: float
    raw_residual_p95_mm: float
    grid: RoughnessGrid
    surface_distances_mm: FloatArray
    variogram_bins_um: FloatArray
    variogram_counts: IntArray
    config: RoughnessConfig

    def metrics_dict(self) -> dict[str, float | int]:
        return {
            "sa_um": self.sa_um,
            "sq_um": self.sq_um,
            "svr_um": self.svr_um,
            "points": self.points,
            "cropped_points": self.cropped_points,
            "grid_width": int(self.grid.raw.shape[1]),
            "grid_height": int(self.grid.raw.shape[0]),
            "grid_pitch_mm": float(self.config.grid_mm),
            "grid_coverage_raw_percent": float(self.grid.valid_raw.mean() * 100.0),
            "grid_coverage_filled_percent": float(self.grid.valid_filled.mean() * 100.0),
            "short_cutoff_mm": float(self.config.short_cutoff_mm),
            "long_cutoff_mm": float(self.config.long_cutoff_mm),
            "mesh_resolution_mm": float(self.config.mesh_resolution_mm),
            "mesh_smoothing_mm": float(self.config.mesh_smoothing_mm),
            "max_svr_points": int(self.config.max_svr_points),
        }

    def save_metrics_json(self, path: str | Path) -> None:
        save_metrics_json(self, path)

    def save_grid_npz(self, path: str | Path) -> None:
        save_grid_npz(self, path)


def save_grid_npz(result: RoughnessResult, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        Path(path),
        grid_raw=result.grid.raw,
        grid_filled=result.grid.filled,
        grid_filtered=result.grid.filtered,
        valid_raw=result.grid.valid_raw,
        valid_filled=result.grid.valid_filled,
        grid_origin=result.grid.origin,
        plane_centroid=result.plane.centroid,
        plane_normal=result.plane.normal,
        plane_x_axis=result.plane.x_axis,
        plane_y_axis=result.plane.y_axis,
    )


def save_metrics_json(result: RoughnessResult, path: str | Path) -> None:
    metrics_path = Path(path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result.metrics_dict(), indent=2) + "\n")


def format_report(result: RoughnessResult) -> str:
    lines = [
        f"input points: {result.points}",
        f"cropped points: {result.cropped_points}",
        f"plane centroid mm: {result.plane.centroid}",
        f"plane normal: {result.plane.normal}",
        (
            "raw residual mm: "
            f"std={result.raw_residual_std_mm:.6f} "
            f"p05={result.raw_residual_p05_mm:.6f} "
            f"p95={result.raw_residual_p95_mm:.6f}"
        ),
        (
            f"grid: {result.grid.raw.shape[1]} x {result.grid.raw.shape[0]} cells, "
            f"pitch={result.config.grid_mm:.4f} mm"
        ),
        (
            f"grid coverage: raw={result.grid.valid_raw.mean() * 100:.1f}% "
            f"filled={result.grid.valid_filled.mean() * 100:.1f}%"
        ),
        (
            f"filters: short_cutoff={result.config.short_cutoff_mm:g} mm "
            f"long_cutoff={result.config.long_cutoff_mm:g} mm"
        ),
        "",
        "Roughness:",
        f"  Sa  = {result.sa_um:.3f} um",
        f"  Sq  = {result.sq_um:.3f} um",
        f"  Svr = {result.svr_um:.3f} um",
        "",
        "Variogram bins:",
    ]
    for idx, value in enumerate(result.variogram_bins_um):
        lo = idx * result.config.svr_span_mm
        hi = (idx + 1) * result.config.svr_span_mm
        if np.isfinite(value):
            lines.append(f"  {lo:.3f}-{hi:.3f} mm: {value:.3f} um  pairs={result.variogram_counts[idx]}")
        else:
            lines.append(f"  {lo:.3f}-{hi:.3f} mm: no pairs")
    return "\n".join(lines)