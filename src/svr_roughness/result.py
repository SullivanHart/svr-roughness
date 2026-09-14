from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
class RoughnessResult:
    """Standardized ASTM WK92969 areal surface roughness result."""

    # ── Primary Metrology Metrics ──
    svr_um: float
    sa_um: float
    sq_um: float
    noise_floor_um: float = 0.0
    svr_raw_um: float = 0.0

    # ── Point Cloud Geometry & Counts ──
    points: int = 0
    processed_points: int = 0
    plane: PlaneFit = None

    # ── 2D Filtered Roughness Grid ──
    grid: FloatArray = None
    grid_pitch_mm: float = 0.20
    grid_origin_mm: FloatArray = None

    # ── Variogram Curve ──
    variogram_bins_um: FloatArray = None
    variogram_counts: IntArray = None
    config: RoughnessConfig = None

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
        if self.plane is None or len(self.plane.coords) == 0:
            return float(self.grid.shape[1] * self.grid_pitch_mm) if self.grid is not None else 0.0
        return float(self.plane.coords[:, 0].max() - self.plane.coords[:, 0].min())

    @property
    def patch_height_mm(self) -> float:
        """Height of the surface patch along the fitted plane Y-axis in mm."""
        if self.plane is None or len(self.plane.coords) == 0:
            return float(self.grid.shape[0] * self.grid_pitch_mm) if self.grid is not None else 0.0
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

    def heatmap(self, radius_mm: float = 5.0) -> FloatArray:
        """Compute the 2D local Svr spatial roughness heatmap (in µm) on demand."""
        from .algorithm import compute_heatmap_grid

        if self.grid is None:
            raise ValueError("No grid available to compute heatmap")
        return compute_heatmap_grid(self.grid, self.grid_pitch_mm, radius_mm)

    def comparator_equivalents(self) -> dict[str, str]:
        """Convert S_VR to approximate equivalent comparator visual grades per Appendices X1-X3."""
        svr = self.svr_mm

        # SCRATA Comparator (A802) - Appendix X2
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

    def to_dict(self) -> dict[str, Any]:
        """Return a clean JSON-serializable dictionary of metrics and metadata."""
        grid_shape = [int(self.grid.shape[0]), int(self.grid.shape[1])] if self.grid is not None else [0, 0]
        return {
            "svr_um": round(self.svr_um, 3),
            "sa_um": round(self.sa_um, 3),
            "sq_um": round(self.sq_um, 3),
            "svr_raw_um": round(self.svr_raw_um, 3),
            "noise_floor_um": round(self.noise_floor_um, 3),
            "points": int(self.points),
            "processed_points": int(self.processed_points),
            "grid_shape": grid_shape,
            "grid_pitch_mm": float(self.grid_pitch_mm),
            "variogram_bins_um": [round(float(v), 4) for v in self.variogram_bins_um] if self.variogram_bins_um is not None else [],
            "variogram_counts": [int(c) for c in self.variogram_counts] if self.variogram_counts is not None else [],
            "comparators": self.comparator_equivalents(),
        }

    def metrics_dict(self) -> dict[str, Any]:
        """Backward-compatible alias for to_dict()."""
        return self.to_dict()

    def save_metrics_json(self, path: str | Path) -> None:
        save_metrics_json(self, path)

    def save_grid_npz(self, path: str | Path) -> None:
        save_grid_npz(self, path)

    def __repr__(self) -> str:
        nf_str = f", noise floor: {self.noise_floor_um:.2f} µm" if self.noise_floor_um > 0 else ""
        grid_info = f"{self.grid.shape[1]} × {self.grid.shape[0]} cells (pitch: {self.grid_pitch_mm:.3f} mm)" if self.grid is not None else "None"
        return (
            f"RoughnessResult(\n"
            f"  Svr = {self.svr_um:.2f} µm (raw: {self.svr_raw_um:.2f} µm{nf_str})\n"
            f"  Sa  = {self.sa_um:.2f} µm\n"
            f"  Sq  = {self.sq_um:.2f} µm\n"
            f"  Points: {self.points:,} (processed: {self.processed_points:,})\n"
            f"  Grid:   {grid_info}\n"
            f")"
        )


def save_grid_npz(result: RoughnessResult, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        Path(path),
        grid=result.grid,
        grid_pitch_mm=result.grid_pitch_mm,
        grid_origin=result.grid_origin_mm,
        plane_centroid=result.plane.centroid if result.plane else np.zeros(3),
        plane_normal=result.plane.normal if result.plane else np.zeros(3),
        plane_x_axis=result.plane.x_axis if result.plane else np.zeros(3),
        plane_y_axis=result.plane.y_axis if result.plane else np.zeros(3),
    )


def save_metrics_json(result: RoughnessResult, path: str | Path) -> None:
    metrics_path = Path(path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result.to_dict(), indent=2) + "\n")


def format_report(result: RoughnessResult, target_svr_mm: float | None = None) -> str:
    """Format a human-readable report per ASTM WK92969 §10."""
    eval_length_mm = result.config.svr_points * result.config.svr_span_mm if result.config else 5.0
    patch_w = result.patch_width_mm
    patch_h = result.patch_height_mm
    density = result.point_density_pts_mm2
    spacing = result.average_point_spacing_mm

    patch_valid = patch_w >= 50.0 and patch_h >= 50.0
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
    if result.noise_floor_um > 0:
        lines.append(f"  (Dynamic Noise Floor: {result.noise_floor_um:.2f} µm, Raw S_VR: {result.svr_raw_um:.2f} µm)")

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
            f"  {eval_length_mm:.1f} mm",
            "",
            "§10.1.3  Distance Bucket Size",
            f"  {result.config.svr_span_mm if result.config else 0.5} mm",
            "",
            "§10.1.4  Cutoff Wavelengths",
            f"  Short (λs): {result.config.short_cutoff_mm:g} mm" if result.config else "  Short: N/A",
            f"  Long  (λc): {result.config.long_cutoff_mm:g} mm" if result.config else "  Long: N/A",
            "",
            "───────────────────────────────────────────────────────",
            "  Surface Patch & Sensor Validation",
            "───────────────────────────────────────────────────────",
            f"  Patch Dimensions (§3.1.5): {patch_w:.1f} × {patch_h:.1f} mm (Area: {result.patch_area_mm2:.0f} mm²)",
            f"  Patch Size Status:         {'VALID (>= 50x50 mm)' if patch_valid else 'WARNING (< 50x50 mm recommended)'}",
            f"  Raw Points:                {result.points:,}",
            f"  Processed Points:          {result.processed_points:,}",
            f"  Avg Point Spacing (§8.2):  {spacing:.3f} mm (~{density:.1f} pts/mm²)",
            f"  Point Density Status:      {'VALID (<= 0.20 mm spacing)' if density_valid else 'WARNING (> 0.20 mm required)'}",
            f"  Grid:                      {result.grid.shape[1]} × {result.grid.shape[0]} cells, pitch={result.grid_pitch_mm:.4f} mm" if result.grid is not None else "  Grid: None",
            "",
            "───────────────────────────────────────────────────────",
            "  Visual Comparator Equivalents (Appendices X1-X3)",
            "───────────────────────────────────────────────────────",
        ]
    )

    for std_name, grade in result.comparator_equivalents().items():
        lines.append(f"  {std_name:<24}: {grade}")

    if result.plane is not None:
        lines.extend(
            [
                "",
                "───────────────────────────────────────────────────────",
                "  Plane & Geometry Alignment",
                "───────────────────────────────────────────────────────",
                f"  Plane centroid: {result.plane.centroid}",
                f"  Plane normal:   {result.plane.normal}",
            ]
        )

    if result.variogram_bins_um is not None:
        lines.extend(
            [
                "",
                "───────────────────────────────────────────────────────",
                "  Variogram Bins",
                "───────────────────────────────────────────────────────",
            ]
        )
        span = result.config.svr_span_mm if result.config else 0.5
        for idx, value in enumerate(result.variogram_bins_um):
            lo = idx * span
            hi = (idx + 1) * span
            if np.isfinite(value):
                pairs_str = f"  pairs={result.variogram_counts[idx]}" if result.variogram_counts is not None and idx < len(result.variogram_counts) else ""
                lines.append(f"  {lo:.3f}–{hi:.3f} mm: {value:.3f} µm ({value / 1000.0:.4f} mm){pairs_str}")
            else:
                lines.append(f"  {lo:.3f}–{hi:.3f} mm: no pairs")

    return "\n".join(lines)
