from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.ndimage import convolve, distance_transform_edt, gaussian_filter

from .config import RoughnessConfig
from .result import BoolArray, FloatArray, PlaneFit


def fit_plane_basis(points: npt.ArrayLike) -> PlaneFit:
    points_array = np.asarray(points, dtype=np.float64)
    centroid = points_array.mean(axis=0)
    centered = points_array - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    x_axis = vh[0]
    x_axis -= normal * np.dot(x_axis, normal)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    coords = np.column_stack((centered @ x_axis, centered @ y_axis, centered @ normal))
    return PlaneFit(centroid=centroid, normal=normal, x_axis=x_axis, y_axis=y_axis, coords=coords)


def crop_points(coords: FloatArray, config: RoughnessConfig) -> tuple[FloatArray, BoolArray]:
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


def grid_residuals(
    coords: FloatArray,
    grid_mm: float,
    min_points_per_cell: int,
) -> tuple[FloatArray, BoolArray, FloatArray]:
    if grid_mm <= 0:
        raise ValueError("grid_mm must be greater than zero")
    if min_points_per_cell < 1:
        raise ValueError("min_points_per_cell must be at least 1")
    if len(coords) == 0:
        raise ValueError("No points remain after cropping")

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    x0, x1 = x.min(), x.max()
    y0, y1 = y.min(), y.max()

    cols = max(1, int(np.ceil((x1 - x0) / grid_mm)) + 1)
    rows = max(1, int(np.ceil((y1 - y0) / grid_mm)) + 1)

    ix = np.clip(np.floor((x - x0) / grid_mm).astype(int), 0, cols - 1)
    iy = np.clip(np.floor((y - y0) / grid_mm).astype(int), 0, rows - 1)

    linear_indices = iy * cols + ix
    cell_count = rows * cols

    counts = np.bincount(linear_indices, minlength=cell_count).reshape((rows, cols))
    sums = np.bincount(linear_indices, weights=z, minlength=cell_count).reshape((rows, cols))

    valid = counts >= min_points_per_cell
    grid = np.full((rows, cols), np.nan, dtype=np.float64)
    grid[valid] = sums[valid] / counts[valid]

    origin = np.array([x0, y0, grid_mm], dtype=np.float64)
    return grid, valid, origin


def fill_holes_neighbor_mean(grid: FloatArray, valid: BoolArray, max_passes: int = 5) -> tuple[FloatArray, BoolArray]:
    filled = grid.copy()
    valid_mask = valid.copy()
    rows, cols = grid.shape

    for _ in range(max_passes):
        missing = ~valid_mask
        if not np.any(missing):
            break

        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float64)
        valid_float = valid_mask.astype(np.float64)
        values_zeroed = np.nan_to_num(filled, nan=0.0)

        neighbor_sums = convolve(values_zeroed, kernel, mode="constant", cval=0.0)
        neighbor_counts = convolve(valid_float, kernel, mode="constant", cval=0.0)

        fillable = missing & (neighbor_counts > 0)
        if not np.any(fillable):
            break

        filled[fillable] = neighbor_sums[fillable] / neighbor_counts[fillable]
        valid_mask[fillable] = True

    return filled, valid_mask


def apply_filters(
    grid_filled: FloatArray,
    valid_mask: BoolArray,
    grid_mm: float,
    short_cutoff_mm: float,
    long_cutoff_mm: float,
) -> FloatArray:
    output = grid_filled.copy()
    if not np.any(valid_mask):
        return output

    if short_cutoff_mm > 0:
        sigma_short = (short_cutoff_mm / grid_mm) / (2.0 * np.pi)
        if sigma_short > 0:
            lowpass = gaussian_filter(output, sigma=sigma_short, mode="nearest")
            output = output - lowpass

    if long_cutoff_mm > 0:
        sigma_long = (long_cutoff_mm / grid_mm) / (2.0 * np.pi)
        if sigma_long > 0:
            form = gaussian_filter(output, sigma=sigma_long, mode="nearest")
            output = output - form

    return output


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

