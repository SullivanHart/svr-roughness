from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

Range = tuple[float, float]


@dataclass(frozen=True)
class RoughnessConfig:
    """Configuration for roughness analysis.

    Input point coordinates are assumed to be millimeters. Roughness metrics are
    reported in micrometers.

    Default values match the canonical SurfInspect/PC_svr2 Settings.csv.
    """

    grid_mm: float = 0.20
    x_range: Range | None = None
    y_range: Range | None = None
    z_range: Range | None = None
    percentile_crop: float = 0.0
    min_points_per_cell: int = 1
    max_hole_passes: int = 8
    long_cutoff_mm: float = 25.0
    short_cutoff_mm: float = 1.0
    svr_points: int = 10
    svr_span_mm: float = 0.5
    mesh_resolution_mm: float = 1.0
    mesh_smoothing_mm: float = 0.0
    max_svr_points: int = 0
    statistical_filter: bool = False
    statistical_mean_k: int = 6
    statistical_stddev: float = 3.0
    # Select the recovered Cloud-Viewer PCA-aligned 1 mm Gaussian mesh path.
    gaussian_mesh: bool = True

    @classmethod
    def surfinspect_defaults(cls) -> RoughnessConfig:
        """Return a config matching the canonical SurfInspect/PC_svr2 defaults."""
        return cls()

    @classmethod
    def astm_standard_defaults(cls) -> RoughnessConfig:
        """Return a config strictly enforcing the ASTM WK92969 standard parameters.

        Note: Appendices X1.3, X2.3, and X3.3 of the draft standard contain a known
        unit-conversion typo for cutoffs (0.001 mm and 0.025 mm). The values used
        here match the intended 1 mm and 25 mm specified in Section 9.1.
        """
        return cls(
            grid_mm=0.20,
            short_cutoff_mm=1.0,
            long_cutoff_mm=25.0,
            svr_span_mm=0.5,
            svr_points=10,
        )


def parse_range(text: str | None) -> Range | None:
    if text is None:
        return None
    lo, hi = text.split(":", 1)
    return float(lo), float(hi)


def coerce_range(value: str | Sequence[float] | None) -> Range | None:
    if value is None:
        return None
    if isinstance(value, str):
        return parse_range(value)
    lo, hi = value
    return float(lo), float(hi)


def config_from_values(
    *,
    grid_mm: float = 0.30,
    x_range: str | Range | None = None,
    y_range: str | Range | None = None,
    z_range: str | Range | None = None,
    percentile_crop: float = 0.0,
    min_points_per_cell: int = 1,
    max_hole_passes: int = 8,
    long_cutoff_mm: float = 0.0,
    short_cutoff_mm: float = 0.0,
    svr_points: int = 1,
    svr_span_mm: float = 0.5,
    mesh_resolution_mm: float = 1.0,
    max_svr_points: int = 0,
    mesh_smoothing_mm: float = 0.0,
    statistical_filter: bool = False,
    statistical_mean_k: int = 6,
    statistical_stddev: float = 3.0,
    gaussian_mesh: bool = True,
) -> RoughnessConfig:
    return RoughnessConfig(
        grid_mm=grid_mm,
        x_range=coerce_range(x_range),
        y_range=coerce_range(y_range),
        z_range=coerce_range(z_range),
        percentile_crop=percentile_crop,
        min_points_per_cell=min_points_per_cell,
        max_hole_passes=max_hole_passes,
        long_cutoff_mm=long_cutoff_mm,
        short_cutoff_mm=short_cutoff_mm,
        svr_points=svr_points,
        svr_span_mm=svr_span_mm,
        mesh_resolution_mm=mesh_resolution_mm,
        mesh_smoothing_mm=mesh_smoothing_mm,
        max_svr_points=max_svr_points,
        statistical_filter=statistical_filter,
        statistical_mean_k=statistical_mean_k,
        statistical_stddev=statistical_stddev,
        gaussian_mesh=gaussian_mesh,
    )
