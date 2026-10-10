"""Surface roughness analysis for scanner point clouds."""

from ._version import version as __version__
from .algorithm import (
    PurePythonResult,
    SparseScanError,
    analyze_pure_python,
    compute_heatmap_grid,
    interpolate_heatmap_at_points,
)
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
from .io import load_delimited, load_obj, load_pcd, load_ply, load_points, load_stl_vertices
from .result import PlaneFit, RoughnessResult, format_report, save_grid_npz, save_metrics_json

__all__ = [
    "DecompositionConfig",
    "ObjectRoughnessResult",
    "PlaneFit",
    "PurePythonResult",
    "RoughnessConfig",
    "RoughnessResult",
    "SparseScanError",
    "SurfacePatchResult",
    "__version__",
    "analyze_file",
    "analyze_object",
    "analyze_ply",
    "analyze_points",
    "analyze_pure_python",
    "compute_heatmap_grid",
    "interpolate_heatmap_at_points",
    "compute_roughness",
    "compute_roughness_from_ply",
    "decompose_3d_object",
    "format_report",
    "is_planar_surface",
    "load_delimited",
    "load_obj",
    "load_pcd",
    "load_ply",
    "load_points",
    "load_stl_vertices",
    "save_grid_npz",
    "save_metrics_json",
]
