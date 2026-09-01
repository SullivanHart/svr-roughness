"""Surface roughness analysis for scanner point clouds."""

__version__ = "0.4.21"


from .analyze import (
    analyze_file,
    analyze_ply,
    analyze_points,
    compute_roughness,
    compute_roughness_from_ply,
)
from .config import RoughnessConfig
from .io import load_ascii_ply, load_delimited, load_obj, load_pcd, load_ply, load_points, load_stl_vertices
from .result import PlaneFit, RoughnessGrid, RoughnessResult, format_report

from ._display_grid import svr_map

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
    "format_report",
    "load_ascii_ply",
    "load_delimited",
    "load_obj",
    "load_pcd",
    "load_ply",
    "load_points",
    "load_stl_vertices",
    "svr_map",
    "__version__",
]