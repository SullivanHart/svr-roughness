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
    svr_map: FloatArray | None = None


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

    @property
    def svr_mm(self) -> float:
        """S_VR in millimeters, as defined by ASTM WK92969 §3.1.3."""
        return self.svr_um / 1000.0

    @property
    def svr_in(self) -> float:
        """S_VR in inches (1 mm = 0.0393701 in)."""
        return self.svr_mm / 25.4

    @property
    def patch_width_mm(self) -> float:
        """Width of the surface patch along the fitted plane X-axis in mm."""
        return float(self.plane.coords[:, 0].max() - self.plane.coords[:, 0].min())

    @property
    def patch_height_mm(self) -> float:
        """Height of the surface patch along the fitted plane Y-axis in mm."""
        return float(self.plane.coords[:, 1].max() - self.plane.coords[:, 1].min())

    @property
    def patch_area_mm2(self) -> float:
        """Estimated surface area of the patch in mm²."""
        return self.patch_width_mm * self.patch_height_mm

    @property
    def point_density_pts_mm2(self) -> float:
        """Average point density of the raw cloud in points per mm²."""
        area = self.patch_area_mm2
        return (self.points / area) if area > 0 else 0.0

    @property
    def average_point_spacing_mm(self) -> float:
        """Estimated average point spacing in mm."""
        density = self.point_density_pts_mm2
        return float(1.0 / np.sqrt(density)) if density > 0 else 0.0

    def comparator_equivalents(self) -> dict[str, str]:
        """Convert S_VR to approximate equivalent comparator visual grades per Appendices X1-X3.

        ASTM WK92969 references:
          - Appendix X1: GAR C-9 Cast Microfinish Comparator
          - Appendix X2: SCRATA Standard (ASTM A802)
          - Appendix X3: ACI Surface Indicator Scale
        """
        svr = self.svr_mm

        # SCRATA Comparator (A802) - Appendix X2
        # A1: 0.0264 mm, A2: 0.0448 mm, A3: 0.0630 mm, A4: 0.1315 mm
        if svr < 0.0214:
            scrata = "< A1 (Finer than 0.026 mm)"
        elif svr <= 0.0356:
            scrata = "A1 (Nominal ~0.026 mm)"
        elif svr <= 0.0539:
            scrata = "A2 (Nominal ~0.045 mm)"
        elif svr <= 0.0973:
            scrata = "A3 (Nominal ~0.063 mm)"
        elif svr <= 0.1500:
            scrata = "A4 (Nominal ~0.132 mm)"
        else:
            scrata = "> A4 (Rougher than 0.132 mm)"

        # GAR C-9 Comparator - Appendix X1
        # 200: 0.0092, 300: 0.0163, 420: 0.0187, 560: 0.0274, 720: 0.0585, 900: 0.0711 mm
        if svr < 0.0089:
            gar = "< C-9 200"
        elif svr <= 0.0128:
            gar = "C-9 200 (Nominal ~0.009 mm)"
        elif svr <= 0.0175:
            gar = "C-9 300 (Nominal ~0.016 mm)"
        elif svr <= 0.0231:
            gar = "C-9 420 (Nominal ~0.019 mm)"
        elif svr <= 0.0430:
            gar = "C-9 560 (Nominal ~0.027 mm)"
        elif svr <= 0.0648:
            gar = "C-9 720 (Nominal ~0.059 mm)"
        elif svr <= 0.0850:
            gar = "C-9 900 (Nominal ~0.071 mm)"
        else:
            gar = "> C-9 900"

        # ACI Surface Indicator Scale (SIS) - Appendix X3
        # SIS-1: 0.0094, SIS-2: 0.0199, SIS-3: 0.0245, SIS-4: 0.0801 mm
        if svr < 0.0088:
            aci = "< SIS-1"
        elif svr <= 0.0147:
            aci = "SIS-1 (Nominal ~0.009 mm)"
        elif svr <= 0.0222:
            aci = "SIS-2 (Nominal ~0.020 mm)"
        elif svr <= 0.0523:
            aci = "SIS-3 (Nominal ~0.025 mm)"
        elif svr <= 0.1000:
            aci = "SIS-4 (Nominal ~0.080 mm)"
        else:
            aci = "> SIS-4"

        return {
            "SCRATA (ASTM A802)": scrata,
            "GAR C-9": gar,
            "ACI SIS": aci,
        }

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


def format_report(result: RoughnessResult, target_svr_mm: float | None = None) -> str:
    """Format a human-readable report per ASTM WK92969 §10.

    §10.1 required fields:
      - 10.1.1  Test result (S_VR)
      - 10.1.2  Evaluation length
      - 10.1.3  Distance bucket size
      - 10.1.4  Cutoff wavelengths
      - 10.1.5  Pre-processing parameters
    """
    eval_length_mm = result.config.svr_points * result.config.svr_span_mm
    patch_w = result.patch_width_mm
    patch_h = result.patch_height_mm
    density = result.point_density_pts_mm2
    spacing = result.average_point_spacing_mm

    # ASTM WK92969 §3.1.5 recommendation: 50x50 mm or larger
    patch_valid = patch_w >= 50.0 and patch_h >= 50.0
    # ASTM WK92969 §8.2 requirement: Point density of 0.2 mm or better required
    density_valid = spacing <= 0.20

    lines = [
        "═══════════════════════════════════════════════════════",
        "  ASTM WK92969 Surface Variogram Roughness (S_VR) Report",
        "═══════════════════════════════════════════════════════",
        "",
        "§10.1.1  Test Result",
        f"  S_VR = {result.svr_mm:.4f} mm  ({result.svr_um:.3f} µm / {result.svr_in:.5f} in)",
        f"  Sa   = {result.sa_um:.3f} µm",
        f"  Sq   = {result.sq_um:.3f} µm",
    ]

    if target_svr_mm is not None and target_svr_mm > 0:
        passed = result.svr_mm <= target_svr_mm
        status_str = "PASS (Within specification)" if passed else "FAIL (Exceeds specification)"
        lines.extend(
            [
                f"  Purchaser Spec Target: <= {target_svr_mm:.4f} mm",
                f"  Status (§5.3):        {status_str}",
            ]
        )

    lines.extend(
        [
            "",
            "§10.1.2  Evaluation Length",
            f"  {eval_length_mm:.1f} mm  ({result.config.svr_points} bins × {result.config.svr_span_mm} mm)",
            "",
            "§10.1.3  Distance Bucket Size",
            f"  {result.config.svr_span_mm} mm",
            "",
            "§10.1.4  Cutoff Wavelengths",
            f"  Short (λs): {result.config.short_cutoff_mm:g} mm",
            f"  Long  (λc): {result.config.long_cutoff_mm:g} mm",
            "",
            "§10.1.5  Pre-processing Parameters",
            f"  Downsampling:   {result.config.grid_mm} mm",
            f"  Outlier filter: {f'ON (k={result.config.statistical_mean_k}, σ={result.config.statistical_stddev})' if result.config.statistical_filter else 'OFF'}",
            f"  Gaussian mesh:  {'ON (ISO 16610-61)' if result.config.gaussian_mesh else 'OFF'}",
            "",
            "───────────────────────────────────────────────────────",
            "  Surface Patch & Sensor Validation",
            "───────────────────────────────────────────────────────",
            f"  Patch Dimensions (§3.1.5): {patch_w:.1f} × {patch_h:.1f} mm (Area: {result.patch_area_mm2:.0f} mm²)",
            f"  Patch Size Status:         {'VALID (>= 50x50 mm)' if patch_valid else 'WARNING (< 50x50 mm recommended)'}",
            f"  Raw Points:                {result.points:,}",
            f"  Processed Points:          {result.cropped_points:,}",
            f"  Avg Point Spacing (§8.2):  {spacing:.3f} mm (~{density:.1f} pts/mm²)",
            f"  Point Density Status:      {'VALID (<= 0.20 mm spacing)' if density_valid else 'WARNING (> 0.20 mm required)'}",
            "  Scanner Accuracy (§7.1):   Recommended < 0.076 mm (0.0030 in)",
            f"  Grid:                      {result.grid.raw.shape[1]} × {result.grid.raw.shape[0]} cells, pitch={result.config.grid_mm:.4f} mm",
            f"  Grid Coverage:             {result.grid.valid_filled.mean() * 100:.1f}%",
            "",
            "───────────────────────────────────────────────────────",
            "  Visual Comparator Equivalents (Appendices X1-X3)",
            "───────────────────────────────────────────────────────",
        ]
    )

    for std_name, grade in result.comparator_equivalents().items():
        lines.append(f"  {std_name:<24}: {grade}")

    lines.extend(
        [
            "",
            "───────────────────────────────────────────────────────",
            "  Plane & Geometry Alignment",
            "───────────────────────────────────────────────────────",
            f"  Plane centroid: {result.plane.centroid}",
            f"  Plane normal:   {result.plane.normal}",
            (
                f"  Raw residual mm: "
                f"std={result.raw_residual_std_mm:.6f} "
                f"p05={result.raw_residual_p05_mm:.6f} "
                f"p95={result.raw_residual_p95_mm:.6f}"
            ),
            "",
            "───────────────────────────────────────────────────────",
            "  Variogram Bins (Evaluation Length 0 to 5.0 mm)",
            "───────────────────────────────────────────────────────",
        ]
    )
    for idx, value in enumerate(result.variogram_bins_um):
        lo = idx * result.config.svr_span_mm
        hi = (idx + 1) * result.config.svr_span_mm
        if np.isfinite(value):
            lines.append(
                f"  {lo:.3f}–{hi:.3f} mm: {value:.3f} µm ({value / 1000.0:.4f} mm)  pairs={result.variogram_counts[idx]}"
            )
        else:
            lines.append(f"  {lo:.3f}–{hi:.3f} mm: no pairs")
    return "\n".join(lines)
