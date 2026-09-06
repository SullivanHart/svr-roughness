"""Backward-compatible native-backed roughness module layout."""

from .analyze import analyze_file, analyze_ply, analyze_points, compute_roughness, compute_roughness_from_ply
from .config import RoughnessConfig, config_from_values, parse_range
from .io import load_ascii_ply, load_delimited, load_obj, load_pcd, load_ply, load_points, load_stl_vertices
from .result import (
    PlaneFit,
    RoughnessGrid,
    RoughnessResult,
    format_report,
    save_grid_npz,
    save_metrics_json,
)

__all__ = [
    "PlaneFit",
    "RoughnessConfig",
    "RoughnessGrid",
    "RoughnessResult",
    "analyze_file",
    "analyze_ply",
    "analyze_points",
    "compute_roughness",
    "compute_roughness_from_ply",
    "config_from_values",
    "format_report",
    "load_ascii_ply",
    "load_delimited",
    "load_obj",
    "load_pcd",
    "load_ply",
    "load_points",
    "load_stl_vertices",
    "parse_range",
    "save_grid_npz",
    "save_metrics_json",
]
