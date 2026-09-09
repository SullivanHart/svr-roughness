"""Pure-Python ASTM WK92969 Gaussian areal roughness numerical core.

Replaces the legacy C++ DLL with a fully vectorized NumPy/SciPy engine.
Implements the exact 5-stage pipeline from SurfInspect:
1. Centering & unit adjustment
2. Voxel grid downsampling (fast 1D key-based grouping)
3. PCA best-fit plane alignment (matching PCL PCA convention)
4. 2.5D elevation grid rasterization with boundary cropping & hole filling
5. Dual-pass ISO 16610-61 Gaussian filtration (alpha=0.4697)
6. 2D FFT autocorrelation variogram (Sa, Sq, Svr) and local Svr dispersion map
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class PurePythonResult(NamedTuple):
    sa_um: float
    sq_um: float
    svr_um: float
    processed_points: int
    variogram_bins_um: np.ndarray
    variogram_counts: np.ndarray
    surface_distances_mm: np.ndarray
    grid_width: int
    grid_height: int
    grid_origin_mm: np.ndarray
    grid_z_mm: np.ndarray
    grid_svr_um: np.ndarray
    noise_floor_um: float = 0.0
    svr_raw_um: float = 0.0


def voxel_downsample(points_m: np.ndarray, voxel_size_m: float) -> np.ndarray:
    """Downsample point cloud using 3D voxel grid centroids (matching pcl::VoxelGrid).

    Uses fast 1D key-based grouping for O(N) performance on large point clouds.
    """
    if len(points_m) == 0:
        return points_m

    min_pt = points_m.min(axis=0)
    coords = np.floor((points_m - min_pt) / voxel_size_m).astype(np.int64)
    max_c = coords.max(axis=0) + 1

    stride_y = max_c[2]
    stride_x = max_c[1] * stride_y
    key = coords[:, 0] * stride_x + coords[:, 1] * stride_y + coords[:, 2]

    _, inverse = np.unique(key, return_inverse=True)
    counts = np.bincount(inverse)

    downsampled = np.column_stack([
        np.bincount(inverse, weights=points_m[:, dim]) / counts
        for dim in range(3)
    ])
    return downsampled


def pca_align_plane(points_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align point cloud surface normal to +Z axis using PCA eigendecomposition (matching PCL PCA)."""
    centroid = points_m.mean(axis=0)
    centered = points_m - centroid
    cov = (centered.T @ centered) / max(len(points_m), 1)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort descending: column 0 is max variance (X), 1 is mid (Y), 2 is normal (Z)
    order = np.argsort(eigenvalues)[::-1]
    eig_vecs = eigenvectors[:, order]

    # Ensure right-handed coordinate system
    if np.linalg.det(eig_vecs) < 0:
        eig_vecs[:, 2] = -eig_vecs[:, 2]

    # Rotate points into plane coordinates (col 0 -> X, col 1 -> Y, col 2 -> Z)
    rotated = centered @ eig_vecs

    # Zero-center Z
    z_mean = rotated[:, 2].mean()
    rotated[:, 2] -= z_mean

    return rotated, centroid, eig_vecs


def _crop_grid_boundaries(valid_mask: np.ndarray) -> tuple[int, int, int, int]:
    """Shave invalid boundary borders matching cropPointCloud in cloud_viewer.cpp.

    Returns (y_begin, y_end, x_begin, x_end).
    """
    missing_xy = (~valid_mask).T
    dim_x, dim_y = missing_xy.shape

    missing_data = np.zeros((dim_x, dim_y), dtype=np.uint8)
    missing_data[missing_xy] = 1

    seen = np.zeros((dim_x, dim_y), dtype=bool)
    dr = [0, 1, 0, -1]
    dc = [1, 0, -1, 0]
    r, c, di = 0, 0, 0
    order: list[tuple[int, int]] = []

    for _ in range(dim_x * dim_y):
        order.append((r, c))
        seen[r, c] = True
        cr = r + dr[di]
        cc = c + dc[di]
        if 0 <= cr < dim_x and 0 <= cc < dim_y and not seen[cr, cc]:
            r, c = cr, cc
        else:
            di = (di + 1) % 4
            r += dr[di]
            c += dc[di]

    for r, c in order:
        if missing_data[r, c] == 1:
            if r == 0 or r == dim_x - 1 or c == 0 or c == dim_y - 1:
                missing_data[r, c] = 2
            elif (
                (r > 0 and missing_data[r - 1, c] == 2)
                or (r < dim_x - 1 and missing_data[r + 1, c] == 2)
                or (c > 0 and missing_data[r, c - 1] == 2)
                or (c < dim_y - 1 and missing_data[r, c + 1] == 2)
            ):
                missing_data[r, c] = 2

    x_begin, x_end = 0, dim_x
    y_begin, y_end = 0, dim_y
    changed = True
    while changed:
        if x_begin >= x_end or y_begin >= y_end:
            return 0, 0, 0, 0
        changed = False
        for x in range(x_begin, x_end):
            if missing_data[x, y_begin] == 2:
                y_begin += 1
                changed = True
                break
        for y in range(y_begin, y_end):
            if missing_data[x_end - 1, y] == 2:
                x_end -= 1
                changed = True
                break
        for x in range(x_begin, x_end):
            if missing_data[x, y_end - 1] == 2:
                y_end -= 1
                changed = True
                break
        for y in range(y_begin, y_end):
            if missing_data[x_begin, y] == 2:
                x_begin += 1
                changed = True
                break

    return y_begin, y_end, x_begin, x_end


def rasterize_elevation_grid(
    rotated_pts_m: np.ndarray,
    pitch_m: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Rasterize 3D points onto regular 2D elevation grid using bilinear scatter interpolation.

    Matches createDenseGridPointCloud and cropPointCloud in cloud_viewer.cpp.
    """
    min_x, min_y = float(rotated_pts_m[:, 0].min()), float(rotated_pts_m[:, 1].min())
    max_x, max_y = float(rotated_pts_m[:, 0].max()), float(rotated_pts_m[:, 1].max())

    steps_x = int(math.ceil((max_x - min_x) / pitch_m)) + 1
    steps_y = int(math.ceil((max_y - min_y) / pitch_m)) + 1

    inv_pitch = 1.0 / pitch_m
    ux = (rotated_pts_m[:, 0] - min_x) * inv_pitch
    uy = (rotated_pts_m[:, 1] - min_y) * inv_pitch

    ix0 = np.clip(np.floor(ux).astype(np.int32), 0, steps_x - 2)
    iy0 = np.clip(np.floor(uy).astype(np.int32), 0, steps_y - 2)
    fx = np.clip(ux - ix0, 0.0, 1.0)
    fy = np.clip(uy - iy0, 0.0, 1.0)

    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy

    grid_sum_zw = np.zeros((steps_y, steps_x), dtype=np.float64)
    grid_sum_w = np.zeros((steps_y, steps_x), dtype=np.float64)

    z = rotated_pts_m[:, 2]
    np.add.at(grid_sum_zw, (iy0, ix0), w00 * z)
    np.add.at(grid_sum_w, (iy0, ix0), w00)

    np.add.at(grid_sum_zw, (iy0, ix0 + 1), w10 * z)
    np.add.at(grid_sum_w, (iy0, ix0 + 1), w10)

    np.add.at(grid_sum_zw, (iy0 + 1, ix0), w01 * z)
    np.add.at(grid_sum_w, (iy0 + 1, ix0), w01)

    np.add.at(grid_sum_zw, (iy0 + 1, ix0 + 1), w11 * z)
    np.add.at(grid_sum_w, (iy0 + 1, ix0 + 1), w11)

    valid = grid_sum_w > 1e-5
    grid_z = np.zeros((steps_y, steps_x), dtype=np.float64)
    grid_z[valid] = grid_sum_zw[valid] / grid_sum_w[valid]
    grid_z[~valid] = np.nan

    # Shave outer missing boundaries (cropPointCloud)
    y_begin, y_end, x_begin, x_end = _crop_grid_boundaries(valid)
    if y_begin >= y_end or x_begin >= x_end:
        raise ValueError("Insufficient contiguous surface area to form a valid roughness grid")
    grid_z = grid_z[y_begin:y_end, x_begin:x_end]
    valid = valid[y_begin:y_end, x_begin:x_end]
    origin_x = min_x + x_begin * pitch_m
    origin_y = min_y + y_begin * pitch_m

    # Fill interior holes (fillHolesInGrid: fast 4-connected 8-pass iterative propagation)
    if np.any(~valid) and np.any(valid):
        filled_z = grid_z.copy()
        current_valid = valid.copy()
        for _ in range(8):
            if np.all(current_valid):
                break
            up = np.roll(filled_z, 1, axis=0)
            down = np.roll(filled_z, -1, axis=0)
            left = np.roll(filled_z, 1, axis=1)
            right = np.roll(filled_z, -1, axis=1)

            up_v = np.roll(current_valid, 1, axis=0)
            down_v = np.roll(current_valid, -1, axis=0)
            left_v = np.roll(current_valid, 1, axis=1)
            right_v = np.roll(current_valid, -1, axis=1)

            up_v[0, :] = False
            down_v[-1, :] = False
            left_v[:, 0] = False
            right_v[:, -1] = False

            sum_neighbors = (
                np.where(up_v, up, 0.0)
                + np.where(down_v, down, 0.0)
                + np.where(left_v, left, 0.0)
                + np.where(right_v, right, 0.0)
            )
            count_neighbors = (
                up_v.astype(np.int32)
                + down_v.astype(np.int32)
                + left_v.astype(np.int32)
                + right_v.astype(np.int32)
            )

            fill_mask = (~current_valid) & (count_neighbors > 0)
            if not np.any(fill_mask):
                break

            filled_z[fill_mask] = sum_neighbors[fill_mask] / count_neighbors[fill_mask]
            current_valid[fill_mask] = True

        if np.any(~current_valid):
            mean_z = np.nanmean(filled_z)
            filled_z[~current_valid] = mean_z
            current_valid[:] = True

        grid_z = filled_z
        valid = current_valid

    return grid_z, valid, origin_x, origin_y


def iso_gaussian_filter_1d_kernel(cutoff_m: float, pitch_m: float) -> np.ndarray:
    """Generate normalized 1D Gaussian filter kernel per ISO 16610-61 (alpha=0.4697)."""
    alpha = 0.4697
    k_size = int(math.ceil(0.5 * cutoff_m / pitch_m)) * 2 + 1
    half = (k_size - 1) // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-np.pi / (alpha**2) * ((x * pitch_m) / cutoff_m) ** 2)
    return kernel / np.sum(kernel)


def _convolve1d_constant(arr: np.ndarray, k_1d: np.ndarray, axis: int) -> np.ndarray:
    """1D convolution matching scipy.ndimage.convolve1d with mode='constant', cval=0.0."""
    k_len = len(k_1d)
    half = k_len // 2
    k_rev = k_1d[::-1]
    if axis == 1:
        padded = np.pad(arr, ((0, 0), (half, half)), mode="constant")
        windows = sliding_window_view(padded, k_len, axis=1)
        return np.tensordot(windows, k_rev, axes=(-1, 0))
    else:
        padded = np.pad(arr, ((half, half), (0, 0)), mode="constant")
        windows = sliding_window_view(padded, k_len, axis=0)
        return np.tensordot(windows, k_rev, axes=(-1, 0))


def _normalized_convolve_2d(arr: np.ndarray, k_1d: np.ndarray) -> np.ndarray:
    """Separable 2D convolution with boundary partial-kernel normalization matching cloud_viewer.cpp."""
    ones = np.ones_like(arr)
    # Pass 1: horizontal (X)
    num = _convolve1d_constant(arr, k_1d, axis=1)
    den = _convolve1d_constant(ones, k_1d, axis=1)
    step1 = num / np.maximum(den, 1e-12)
    # Pass 2: vertical (Y)
    num = _convolve1d_constant(step1, k_1d, axis=0)
    den = _convolve1d_constant(ones, k_1d, axis=0)
    return num / np.maximum(den, 1e-12)


def apply_dual_pass_gaussian_filter(
    grid_z_m: np.ndarray,
    pitch_m: float,
    short_cutoff_m: float,
    long_cutoff_m: float,
) -> np.ndarray:
    """Dual-pass separable areal Gaussian filtration per ASTM WK92969 / ISO 16610-61.

    Matches filterShortLong in cloud_viewer.cpp:
    Pass 1: Short wavelength low-pass (removes high-frequency scanner noise).
    Pass 2: Long wavelength low-pass applied to z_short (extracts form/waviness).
    Roughness profile: roughness_z = z_short - z_long.
    """
    z_short = grid_z_m
    if short_cutoff_m > 0:
        k_short = iso_gaussian_filter_1d_kernel(short_cutoff_m, pitch_m)
        z_short = _normalized_convolve_2d(grid_z_m, k_short)

    if long_cutoff_m > 0:
        k_long = iso_gaussian_filter_1d_kernel(long_cutoff_m, pitch_m)
        z_long = _normalized_convolve_2d(z_short, k_long)
        return z_short - z_long

    return z_short


def _fft_convolve2d_same(in1: np.ndarray, in2: np.ndarray) -> np.ndarray:
    """2D FFT convolution matching scipy.signal.fftconvolve(..., mode='same')."""
    s1 = in1.shape
    s2 = in2.shape
    shape = (s1[0] + s2[0] - 1, s1[1] + s2[1] - 1)
    sp1 = np.fft.rfft2(in1, s=shape)
    sp2 = np.fft.rfft2(in2, s=shape)
    ret = np.fft.irfft2(sp1 * sp2, s=shape)
    f0 = (s2[0] - 1) // 2
    f1 = (s2[1] - 1) // 2
    return ret[f0 : f0 + s1[0], f1 : f1 + s1[1]]


def compute_variogram_and_svr(
    roughness_z_m: np.ndarray,
    pitch_m: float,
    points_on_var: int = 10,
    span_m: float = 0.0005,
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute ASTM WK92969 Sa, Sq, Svr, variogram buckets, and local Svr grid.

    Uses 2D FFT autocorrelation for O(1) pairwise displacement accumulation (~38 ms)
    and 2D FFT convolution for the spatial Svr heatmap grid (~50 ms).
    """
    height, width = roughness_z_m.shape

    # Sa and Sq in micrometers
    sa_um = float(np.mean(np.abs(roughness_z_m)) * 1e6)
    sq_um = float(np.sqrt(np.mean(roughness_z_m**2)) * 1e6)

    # Evaluation length parameter setup
    max_ev_m = points_on_var * span_m
    max_cells = int(math.ceil(max_ev_m / pitch_m))

    z_um = roughness_z_m * 1e6
    pad_h = height + max_cells + 1
    pad_w = width + max_cells + 1

    # Global Variogram via 2D FFT Autocorrelation
    z_fft = np.fft.rfft2(z_um, s=(pad_h, pad_w))
    autocorr = np.fft.irfft2(z_fft * np.conj(z_fft), s=(pad_h, pad_w))
    integral_sq = np.pad(np.cumsum(np.cumsum(z_um**2, axis=0), axis=1), ((1, 0), (1, 0)))

    var_sums = np.zeros(points_on_var, dtype=np.float64)
    var_counts = np.zeros(points_on_var, dtype=np.int64)

    for dy in range(max_cells + 1):
        for dx in range(-max_cells, max_cells + 1):
            if dy == 0 and dx <= 0:
                continue
            dist = math.sqrt((dx * pitch_m) ** 2 + (dy * pitch_m) ** 2)
            b = int(math.floor(dist / span_m))
            if b >= points_on_var:
                continue

            y_src0 = dy
            y_src1 = height
            x_src0 = max(0, dx)
            x_src1 = min(width, width + dx)

            y_dst0 = 0
            y_dst1 = height - dy
            x_dst0 = max(0, -dx)
            x_dst1 = min(width, width - dx)

            s1 = (
                integral_sq[y_src1, x_src1]
                - integral_sq[y_src0, x_src1]
                - integral_sq[y_src1, x_src0]
                + integral_sq[y_src0, x_src0]
            )
            s2 = (
                integral_sq[y_dst1, x_dst1]
                - integral_sq[y_dst0, x_dst1]
                - integral_sq[y_dst1, x_dst0]
                + integral_sq[y_dst0, x_dst0]
            )

            ac_col = dx if dx >= 0 else pad_w + dx
            prod = autocorr[dy, ac_col]

            diff_sq_sum = s1 + s2 - 2.0 * prod
            n_pairs = (height - dy) * (width - abs(dx))

            var_sums[b] += 2.0 * diff_sq_sum
            var_counts[b] += 2 * n_pairs

    var_bins_um = np.zeros(points_on_var, dtype=np.float64)
    valid_bins = var_counts > 0
    var_bins_um[valid_bins] = np.sqrt(var_sums[valid_bins] / (2.0 * var_counts[valid_bins]))

    # Svr is mean of variogram bins (ASTM WK92969 & SurfInspect)
    svr_um = float(np.mean(var_bins_um[valid_bins])) if np.any(valid_bins) else 0.0

    # Local Svr Spatial Heatmap via 2D FFT Convolution
    y_k, x_k = np.ogrid[-max_cells : max_cells + 1, -max_cells : max_cells + 1]
    dist_k = np.sqrt((x_k * pitch_m) ** 2 + (y_k * pitch_m) ** 2)
    kernel = ((dist_k <= max_ev_m) & (dist_k > 0)).astype(np.float64)

    ones = np.ones_like(z_um)
    w_local = _fft_convolve2d_same(ones, kernel)
    conv_z = _fft_convolve2d_same(z_um, kernel)
    conv_z2 = _fft_convolve2d_same(z_um**2, kernel)

    pixel_sum_sq = w_local * (z_um**2) + conv_z2 - 2.0 * z_um * conv_z
    pixel_sum_sq = np.maximum(pixel_sum_sq, 0.0)
    grid_svr_um = np.sqrt(pixel_sum_sq / np.maximum(w_local, 1.0))

    return sa_um, sq_um, svr_um, var_bins_um, var_counts, grid_svr_um


def analyze_pure_python(
    points_xyz_mm: np.ndarray,
    voxel_size_mm: float = 0.20,
    short_cutoff_mm: float = 1.0,
    long_cutoff_mm: float = 25.0,
    variogram_points: int = 10,
    variogram_span_mm: float = 0.5,
    subtract_noise: bool = True,
) -> PurePythonResult:
    """Execute the complete SurfInspect ASTM WK92969 analysis in 100% pure Python."""
    raw_pts = np.asarray(points_xyz_mm, dtype=np.float64)
    if raw_pts.ndim != 2 or raw_pts.shape[1] != 3:
        raise ValueError("Expected an Nx3 point array")

    finite_mask = np.isfinite(raw_pts).all(axis=1)
    pts = raw_pts[finite_mask]
    if len(pts) < 20:
        raise ValueError("SurfInspect engine requires at least 20 valid points")

    # Step 1: Centering & Conversion to meters
    centroid = pts.mean(axis=0)
    pts_m = (pts - centroid) * 0.001
    voxel_size_m = voxel_size_mm * 0.001
    short_cutoff_m = short_cutoff_mm * 0.001
    long_cutoff_m = long_cutoff_mm * 0.001
    span_m = variogram_span_mm * 0.001

    # Step 2: Voxel Grid Downsampling
    downsampled_m = voxel_downsample(pts_m, voxel_size_m)

    # Step 3: Form Removal via PCA Plane Fitting
    rotated_m, _, _ = pca_align_plane(downsampled_m)

    # Step 4: 2.5D Regular Elevation Grid Rasterization
    grid_z_m, valid_mask, origin_x_m, origin_y_m = rasterize_elevation_grid(rotated_m, voxel_size_m)

    # Dynamic instrument noise estimation on elevation grid via 2D discrete Laplacian MAD:
    lap = _fft_convolve2d_same(grid_z_m, np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64))
    valid_lap = valid_mask & np.isfinite(lap)
    if np.any(valid_lap):
        mad = float(np.median(np.abs(lap[valid_lap] - np.median(lap[valid_lap]))))
        noise_floor_um = float((mad / 0.6745 / math.sqrt(20.0)) * 1e6)
    else:
        noise_floor_um = 0.0

    # Step 5: Dual-Pass ISO 16610-61 Gaussian Filtration
    roughness_z_m = apply_dual_pass_gaussian_filter(grid_z_m, voxel_size_m, short_cutoff_m, long_cutoff_m)

    # Step 6: Variogram & Svr Metrology
    sa_um, sq_um, svr_raw_um, var_bins_um, var_counts, grid_svr_um = compute_variogram_and_svr(
        roughness_z_m,
        voxel_size_m,
        variogram_points,
        span_m,
    )

    # Single definitive Svr measurement (with dynamic noise floor subtraction if enabled)
    if subtract_noise and noise_floor_um > 0.0:
        svr_um = float(math.sqrt(max(0.0, svr_raw_um**2 - noise_floor_um**2)))
    else:
        svr_um = svr_raw_um

    height, width = roughness_z_m.shape
    grid_z_mm = roughness_z_m * 1000.0
    surface_distances_mm = grid_z_mm.ravel()

    return PurePythonResult(
        sa_um=sa_um,
        sq_um=sq_um,
        svr_um=svr_um,
        processed_points=len(downsampled_m),
        variogram_bins_um=var_bins_um,
        variogram_counts=var_counts,
        surface_distances_mm=surface_distances_mm,
        grid_width=width,
        grid_height=height,
        grid_origin_mm=np.array([origin_x_m * 1000.0, origin_y_m * 1000.0, voxel_size_mm], dtype=np.float64),
        grid_z_mm=grid_z_mm,
        grid_svr_um=grid_svr_um,
        noise_floor_um=noise_floor_um,
        svr_raw_um=svr_raw_um,
    )

