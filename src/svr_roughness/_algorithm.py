from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt
from scipy.spatial import Delaunay, cKDTree
from scipy.ndimage import distance_transform_edt, gaussian_filter
from typing import Callable

from .config import RoughnessConfig
from .result import BoolArray, FloatArray, IntArray, PlaneFit, RoughnessGrid, RoughnessResult


def clean_points(points: npt.ArrayLike) -> FloatArray:
    points_array = np.asarray(points, dtype=np.float64)
    if points_array.ndim != 2 or points_array.shape[1] != 3:
        raise ValueError("Expected an Nx3 point array")
    points_array = points_array[np.isfinite(points_array).all(axis=1)]
    if len(points_array) < 3:
        raise ValueError("Need at least 3 valid points")
    return points_array


def voxel_downsample(points: FloatArray, leaf_mm: float) -> FloatArray:
    """3D isotropic voxel grid downsampling matching PCL VoxelGrid."""
    if leaf_mm <= 0 or len(points) == 0:
        return points
    voxel_indices = np.floor(points / leaf_mm).astype(np.int64)
    _, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)
    counts = np.bincount(inverse_indices).astype(np.float64)
    sum_x = np.bincount(inverse_indices, weights=points[:, 0])
    sum_y = np.bincount(inverse_indices, weights=points[:, 1])
    sum_z = np.bincount(inverse_indices, weights=points[:, 2])
    return np.column_stack((sum_x / counts, sum_y / counts, sum_z / counts))


def statistical_filter(
    points: FloatArray,
    mean_k: int = 6,
    stddev_multiplier: float = 3.0,
) -> FloatArray:
    """Match SurfInspect's PCL statistical outlier filter defaults."""
    if mean_k < 1 or stddev_multiplier < 0:
        raise ValueError("mean_k must be positive and stddev_multiplier non-negative")
    if len(points) <= mean_k:
        return points
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=mean_k + 1, workers=-1)
    mean_distances = distances[:, 1:].mean(axis=1)
    threshold = mean_distances.mean() + stddev_multiplier * mean_distances.std()
    return points[mean_distances <= threshold]


def fit_plane_basis(points: npt.ArrayLike) -> PlaneFit:
    points_array = clean_points(points)
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
    width = int(np.floor((x1 - x0) / grid_mm)) + 1
    height = int(np.floor((y1 - y0) / grid_mm)) + 1
    ix = np.clip(((x - x0) / grid_mm).astype(np.int32), 0, width - 1)
    iy = np.clip(((y - y0) / grid_mm).astype(np.int32), 0, height - 1)

    sums = np.zeros((height, width), dtype=np.float64)
    counts = np.zeros((height, width), dtype=np.int32)
    np.add.at(sums, (iy, ix), z)
    np.add.at(counts, (iy, ix), 1)
    grid = np.full((height, width), np.nan, dtype=np.float64)
    valid = counts >= min_points_per_cell
    grid[valid] = sums[valid] / counts[valid]
    return grid, valid, np.array((x0, y0, grid_mm), dtype=np.float64)


def fill_holes_neighbor_mean(
    grid: FloatArray,
    valid: BoolArray,
    max_passes: int,
) -> tuple[FloatArray, BoolArray]:
    filled = grid.copy()
    mask = valid.copy()
    kernel = np.ones((3, 3), dtype=np.float64)
    kernel[1, 1] = 0
    for _ in range(max_passes):
        missing = ~mask
        if not missing.any():
            break
        values = np.where(mask, filled, 0.0)
        neighbor_sum = cv2.filter2D(values, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        neighbor_count = cv2.filter2D(mask.astype(np.float64), -1, kernel, borderType=cv2.BORDER_CONSTANT)
        can_fill = missing & (neighbor_count > 0)
        if not can_fill.any():
            break
        filled[can_fill] = neighbor_sum[can_fill] / neighbor_count[can_fill]
        mask[can_fill] = True
    return filled, mask


def gaussian_sigma_from_cutoff(cutoff_mm: float, grid_mm: float) -> float:
    # Cloud-Viewer / PC_svr2 ISO-style kernel uses alpha = 0.4697 as a cutoff scaling constant.
    return max((0.4697 * cutoff_mm) / (np.sqrt(2 * np.pi) * grid_mm), 0.5)


def nan_aware_gaussian(grid: FloatArray, valid: BoolArray, sigma: float) -> FloatArray:
    values = np.where(valid, grid, 0.0).astype(np.float64)
    weights = valid.astype(np.float64)
    blurred_values = cv2.GaussianBlur(
        values,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REPLICATE,
    )
    blurred_weights = cv2.GaussianBlur(
        weights,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REPLICATE,
    )
    return blurred_values / np.maximum(blurred_weights, 1e-9)


def apply_filters(
    grid: FloatArray,
    valid: BoolArray,
    grid_mm: float,
    short_cutoff_mm: float,
    long_cutoff_mm: float,
) -> FloatArray:
    filtered = grid.copy()
    if short_cutoff_mm > 0:
        sigma = gaussian_sigma_from_cutoff(short_cutoff_mm, grid_mm)
        filtered = nan_aware_gaussian(filtered, valid, sigma)
    if long_cutoff_mm > 0:
        sigma = gaussian_sigma_from_cutoff(long_cutoff_mm, grid_mm)
        lowpass = nan_aware_gaussian(filtered, valid, sigma)
        filtered = filtered - lowpass
    return filtered


def sa_sq(grid: FloatArray, valid: BoolArray) -> tuple[float, float]:
    z_um = grid[valid] * 1000.0
    sa = float(np.mean(np.abs(z_um)))
    sq = float(np.sqrt(np.mean(z_um**2)))
    return sa, sq


def svr_grid(
    grid: FloatArray,
    valid: BoolArray,
    grid_mm: float,
    points_on_variogram: int,
    span_mm: float,
) -> tuple[float, FloatArray, IntArray]:
    if points_on_variogram < 1:
        raise ValueError("points_on_variogram must be at least 1")
    if span_mm <= 0:
        raise ValueError("span_mm must be greater than zero")

    max_radius = points_on_variogram * span_mm
    max_cells = int(np.ceil(max_radius / grid_mm))
    sums = np.zeros(points_on_variogram, dtype=np.float64)
    counts = np.zeros(points_on_variogram, dtype=np.int64)

    for dy in range(-max_cells, max_cells + 1):
        for dx in range(-max_cells, max_cells + 1):
            if dx == 0 and dy == 0:
                continue
            if abs(dx) >= grid.shape[1] or abs(dy) >= grid.shape[0]:
                continue
            dist = np.hypot(dx * grid_mm, dy * grid_mm)
            if dist <= 0 or dist > max_radius:
                continue
            bin_idx = int(np.floor(dist / span_mm))
            if bin_idx < 0 or bin_idx >= points_on_variogram:
                continue

            y_src = slice(max(0, -dy), min(grid.shape[0], grid.shape[0] - dy))
            y_dst = slice(max(0, dy), min(grid.shape[0], grid.shape[0] + dy))
            x_src = slice(max(0, -dx), min(grid.shape[1], grid.shape[1] - dx))
            x_dst = slice(max(0, dx), min(grid.shape[1], grid.shape[1] + dx))
            valid_pair = valid[y_src, x_src] & valid[y_dst, x_dst]
            if not valid_pair.any():
                continue
            dz_um = (grid[y_src, x_src][valid_pair] - grid[y_dst, x_dst][valid_pair]) * 1000.0
            sums[bin_idx] += np.sum(dz_um**2)
            counts[bin_idx] += len(dz_um)

    var = np.full(points_on_variogram, np.nan, dtype=np.float64)
    ok = counts > 0
    var[ok] = np.sqrt(sums[ok] / (2.0 * counts[ok]))
    svr = float(np.nanmean(var))
    return svr, var, counts


def mesh_surface_distances(
    coords: FloatArray,
    resolution_mm: float,
    smoothing_mm: float = 0.0,
) -> FloatArray:
    """Approximate a continuous reference mesh and return signed distances."""
    if resolution_mm <= 0:
        raise ValueError("resolution_mm must be greater than zero")
    origin = coords[:, :2].min(axis=0)
    cells = np.floor((coords[:, :2] - origin) / resolution_mm).astype(np.int64)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    cell_coords = np.unique(cells, axis=0)
    vertices = np.column_stack((
        origin[0] + (cell_coords[:, 0] + 0.5) * resolution_mm,
        origin[1] + (cell_coords[:, 1] + 0.5) * resolution_mm,
        np.bincount(inverse, weights=coords[:, 2]) / counts,
    ))
    if smoothing_mm > 0:
        width = int(cells[:, 0].max()) + 1
        height = int(cells[:, 1].max()) + 1
        surface = np.full((height, width), np.nan, dtype=np.float64)
        surface[cell_coords[:, 1], cell_coords[:, 0]] = vertices[:, 2]
        missing = ~np.isfinite(surface)
        if missing.any():
            nearest = distance_transform_edt(missing, return_distances=False, return_indices=True)
            surface[missing] = surface[tuple(nearest[:, missing])]
        sigma = smoothing_mm / resolution_mm
        surface = gaussian_filter(surface, sigma=sigma, mode="nearest")
        vertices[:, 2] = surface[cell_coords[:, 1], cell_coords[:, 0]]
    if len(vertices) < 3:
        raise ValueError("At least 3 points are required to construct a reference surface")
    try:
        triangulation = Delaunay(vertices[:, :2])
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise ValueError("Could not construct a non-degenerate reference surface") from exc

    simplex = triangulation.find_simplex(coords[:, :2])
    distances = np.full(len(coords), np.nan, dtype=np.float64)
    inside = simplex >= 0
    transforms = triangulation.transform[simplex[inside]]
    delta = coords[inside, :2] - transforms[:, 2]
    barycentric = np.einsum("ijk,ik->ij", transforms[:, :2], delta)
    weights = np.column_stack((barycentric, 1.0 - barycentric.sum(axis=1)))
    triangle_vertices = triangulation.simplices[simplex[inside]]
    surface_z = np.sum(vertices[triangle_vertices, 2] * weights, axis=1)
    distances[inside] = coords[inside, 2] - surface_z
    return distances


def svr_surface(
    coords: FloatArray,
    resolution_mm: float,
    points_on_variogram: int,
    span_mm: float,
    max_points: int,
    smoothing_mm: float = 0.0,
) -> tuple[float, FloatArray, IntArray, FloatArray]:
    """Calculate SurfInspect-style Svr from point-to-reference-surface distances."""
    distances = mesh_surface_distances(coords, resolution_mm, smoothing_mm)
    valid = np.isfinite(distances)
    sample = np.flatnonzero(valid)
    if max_points < 0:
        raise ValueError("max_points must be non-negative")
    if max_points and len(sample) > max_points:
        sample = sample[np.linspace(0, len(sample) - 1, max_points, dtype=np.int64)]
    xy = coords[sample, :2]
    values = distances[sample]
    tree = cKDTree(xy)
    pairs = np.asarray(list(tree.query_pairs(points_on_variogram * span_mm)), dtype=np.int64).reshape(-1, 2)
    sums = np.zeros(points_on_variogram, dtype=np.float64)
    counts = np.zeros(points_on_variogram, dtype=np.int64)
    if len(pairs):
        bins = np.floor(np.linalg.norm(xy[pairs[:, 0]] - xy[pairs[:, 1]], axis=1) / span_mm).astype(int)
        keep = bins < points_on_variogram
        squared = ((values[pairs[:, 0]] - values[pairs[:, 1]]) * 1000.0) ** 2
        np.add.at(sums, bins[keep], squared[keep])
        np.add.at(counts, bins[keep], 1)
    bands = np.full(points_on_variogram, np.nan)
    available = counts > 0
    bands[available] = np.sqrt(sums[available] / (2.0 * counts[available]))
    return float(np.nanmean(bands)), bands, counts, distances


def svr_map(
    grid: FloatArray,
    valid: BoolArray,
    grid_mm: float,
    points_on_variogram: int,
    span_mm: float,
) -> FloatArray:
    """Calculate a local Svr value at each valid grid cell.

    The distance bands and pairwise Svr calculation match ``svr_grid``. Each
    cell receives the mean of its available distance-band Svr values.
    """
    if points_on_variogram < 1:
        raise ValueError("points_on_variogram must be at least 1")
    if span_mm <= 0:
        raise ValueError("span_mm must be greater than zero")

    max_cells = int(np.ceil(points_on_variogram * span_mm / grid_mm))
    sums = np.zeros((points_on_variogram,) + grid.shape, dtype=np.float64)
    counts = np.zeros((points_on_variogram,) + grid.shape, dtype=np.int64)

    for dy in range(-max_cells, max_cells + 1):
        for dx in range(-max_cells, max_cells + 1):
            if (dx == 0 and dy == 0) or abs(dx) >= grid.shape[1] or abs(dy) >= grid.shape[0]:
                continue
            distance = np.hypot(dx * grid_mm, dy * grid_mm)
            if distance <= 0 or distance > points_on_variogram * span_mm:
                continue
            bin_idx = int(np.floor(distance / span_mm))
            if bin_idx >= points_on_variogram:
                continue

            y_src = slice(max(0, -dy), min(grid.shape[0], grid.shape[0] - dy))
            y_dst = slice(max(0, dy), min(grid.shape[0], grid.shape[0] + dy))
            x_src = slice(max(0, -dx), min(grid.shape[1], grid.shape[1] - dx))
            x_dst = slice(max(0, dx), min(grid.shape[1], grid.shape[1] + dx))
            valid_pair = valid[y_src, x_src] & valid[y_dst, x_dst]
            difference = (grid[y_src, x_src] - grid[y_dst, x_dst]) * 1000.0
            sum_block = sums[bin_idx, y_src, x_src]
            count_block = counts[bin_idx, y_src, x_src]
            sum_block[valid_pair] += difference[valid_pair] ** 2
            count_block[valid_pair] += 1

    band_values = np.full_like(sums, np.nan)
    available = counts > 0
    band_values[available] = np.sqrt(sums[available] / (2.0 * counts[available]))
    with np.errstate(invalid="ignore"):
        local = np.nanmean(band_values, axis=0)
    local[~valid] = np.nan
    return local


def calculate(
    points: npt.ArrayLike,
    config: RoughnessConfig,
    progress: Callable[[str, float], None] | None = None,
) -> RoughnessResult:
    report = progress or (lambda _label, _value: None)
    report("Preparing point cloud", 0.15)
    points_array = clean_points(points)
    plane = fit_plane_basis(points_array)
    cropped, _ = crop_points(plane.coords, config)
    if config.statistical_filter:
        cropped = statistical_filter(cropped, config.statistical_mean_k, config.statistical_stddev)
        if len(cropped) < 3:
            raise ValueError("Statistical filtering removed too many points")
    grid_raw, valid_raw, grid_origin = grid_residuals(
        cropped,
        config.grid_mm,
        config.min_points_per_cell,
    )
    grid_filled, valid_filled = fill_holes_neighbor_mean(
        grid_raw,
        valid_raw,
        config.max_hole_passes,
    )
    report("Building reference surface", 0.40)
    grid_filtered = apply_filters(
        grid_filled,
        valid_filled,
        config.grid_mm,
        config.short_cutoff_mm,
        config.long_cutoff_mm,
    )

    grid_filtered = grid_filtered - np.nanmean(grid_filtered[valid_filled])
    sa, sq = sa_sq(grid_filtered, valid_filled)
    report("Calculating Svr variogram", 0.70)
    svr, variogram_bins, variogram_counts, surface_distances = svr_surface(
        cropped,
        config.mesh_resolution_mm,
        config.svr_points,
        config.svr_span_mm,
        config.max_svr_points,
        config.mesh_smoothing_mm,
    )

    result = RoughnessResult(
        sa_um=sa,
        sq_um=sq,
        svr_um=svr,
        points=int(len(points_array)),
        cropped_points=int(len(cropped)),
        plane=plane,
        raw_residual_std_mm=float(plane.coords[:, 2].std()),
        raw_residual_p05_mm=float(np.percentile(plane.coords[:, 2], 5)),
        raw_residual_p95_mm=float(np.percentile(plane.coords[:, 2], 95)),
        grid=RoughnessGrid(
            raw=grid_raw,
            filled=grid_filled,
            filtered=grid_filtered,
            valid_raw=valid_raw,
            valid_filled=valid_filled,
            origin=grid_origin,
        ),
        surface_distances_mm=surface_distances,
        variogram_bins_um=variogram_bins,
        variogram_counts=variogram_counts,
        config=config,
    )
    report("Complete", 1.0)
    return result