"""Surface roughness analysis for scanner point clouds."""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.7.1"
from .algorithm import PurePythonResult, analyze_pure_python, compute_heatmap_grid
from .analyze import (
    analyze_file,
    analyze_object,
    analyze_ply,
    analyze_points,
    compute_roughness,
    compute_roughness_from_ply,
)
from .config import RoughnessConfig
from .decomposition import (
    DecompositionConfig,
    ObjectRoughnessResult,
    SurfacePatchResult,
    decompose_3d_object,
    is_planar_surface,
)
from .io import load_ascii_ply, load_delimited, load_obj, load_pcd, load_ply, load_points, load_stl_vertices
from .result import PlaneFit, RoughnessResult, format_report, save_grid_npz, save_metrics_json

__all__ = [
    "DecompositionConfig",
    "ObjectRoughnessResult",
    "PlaneFit",
    "PurePythonResult",
    "RoughnessConfig",
    "RoughnessResult",
    "SurfacePatchResult",
    "__version__",
    "analyze_file",
    "analyze_object",
    "analyze_ply",
    "analyze_points",
    "analyze_pure_python",
    "compute_heatmap_grid",
    "compute_roughness",
    "compute_roughness_from_ply",
    "decompose_3d_object",
    "format_report",
    "is_planar_surface",
    "load_ascii_ply",
    "load_delimited",
    "load_obj",
    "load_pcd",
    "load_ply",
    "load_points",
    "load_stl_vertices",
    "save_grid_npz",
    "save_metrics_json",
]
