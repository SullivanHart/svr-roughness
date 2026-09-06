from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .config import RoughnessConfig
from .result import BoolArray, FloatArray, PlaneFit


def fit_plane_basis(points: npt.ArrayLike) -> PlaneFit:
    """Fit a plane to the point cloud and return a local coordinate system.

    The X-axis is the projection of global X onto the fitted plane, preserving
    the original scan orientation rather than rotating to PCA axes.
    """
    points_array = np.asarray(points, dtype=np.float64)
    centroid = points_array.mean(axis=0)
    centered = points_array - centroid
    # Fast 3x3 covariance eigen-decomposition (O(N) single pass vs O(N*3^2) full SVD)
    cov = (centered.T @ centered) / max(len(points_array) - 1, 1)
    _, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]
    if normal[2] < 0:
        normal = -normal
    normal /= np.linalg.norm(normal)

    # Project global X-axis onto the fitted plane to keep the original scan orientation
    # instead of rotating based on point distribution variance (PCA).
    global_x = np.array([1.0, 0.0, 0.0])
    x_axis = global_x - normal * np.dot(global_x, normal)
    x_axis_norm = np.linalg.norm(x_axis)

    if x_axis_norm < 1e-6:
        global_y = np.array([0.0, 1.0, 0.0])
        x_axis = global_y - normal * np.dot(global_y, normal)
        x_axis /= np.linalg.norm(x_axis)
    else:
        x_axis /= x_axis_norm

    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    coords = np.column_stack((centered @ x_axis, centered @ y_axis, centered @ normal))
    return PlaneFit(centroid=centroid, normal=normal, x_axis=x_axis, y_axis=y_axis, coords=coords)


def crop_points(coords: FloatArray, config: RoughnessConfig) -> tuple[FloatArray, BoolArray]:
    """Crop points by percentile and/or range filters."""
    mask = np.ones(len(coords), dtype=bool)
    if config.percentile_crop > 0:
        p = config.percentile_crop
        xlo, xhi = np.percentile(coords[:, 0], [p, 100 - p])
        ylo, yhi = np.percentile(coords[:, 1], [p, 100 - p])
        mask &= (coords[:, 0] >= xlo) & (coords[:, 0] <= xhi)
        mask &= (coords[:, 1] >= ylo) & (coords[:, 1] <= yhi)
    if config.x_range:
        mask &= (coords[:, 0] >= config.x_range[0]) & (coords[:, 0] <= config.x_range[1])
    if config.y_range:
        mask &= (coords[:, 1] >= config.y_range[0]) & (coords[:, 1] <= config.y_range[1])
    if config.z_range:
        mask &= (coords[:, 2] >= config.z_range[0]) & (coords[:, 2] <= config.z_range[1])
    return coords[mask], mask


def svr_map(
    grid_filtered: FloatArray,
    valid_mask: BoolArray,
    grid_mm: float,
    svr_points: int = 20,
    svr_span_mm: float = 0.5,
) -> FloatArray:
    """Compute local Svr variogram map (in µm) for 2D heatmap rendering."""
    rows, cols = grid_filtered.shape
    output = np.full((rows, cols), np.nan, dtype=np.float64)
    if not np.any(valid_mask) or grid_mm <= 0 or svr_span_mm <= 0:
        return output

    max_dist_mm = svr_points * svr_span_mm
    radius_cells = int(np.ceil(max_dist_mm / grid_mm))

    y_indices, x_indices = np.where(valid_mask)
    for r, c in zip(y_indices, x_indices):
        r_min = max(0, r - radius_cells)
        r_max = min(rows, r + radius_cells + 1)
        c_min = max(0, c - radius_cells)
        c_max = min(cols, c + radius_cells + 1)

        sub_grid = grid_filtered[r_min:r_max, c_min:c_max]
        sub_valid = valid_mask[r_min:r_max, c_min:c_max]

        center_val = grid_filtered[r, c]
        sub_y, sub_x = np.ogrid[r_min - r : r_max - r, c_min - c : c_max - c]
        dists_mm = np.sqrt(sub_y**2 + sub_x**2) * grid_mm

        mask = sub_valid & (dists_mm > 0) & (dists_mm <= max_dist_mm)
        if not np.any(mask):
            continue

        diffs_um = (center_val - sub_grid[mask]) * 1000.0
        output[r, c] = float(np.sqrt(np.mean(diffs_um**2)))

    return output
