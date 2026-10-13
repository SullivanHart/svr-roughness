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
import os
from typing import NamedTuple

import numpy as np
from scipy import ndimage
from scipy.fft import next_fast_len
from scipy.spatial import ConvexHull


class SparseScanError(ValueError):
    """Raised when point spacing is too coarse for the requested grid pitch."""

    def __init__(self, message: str, suggested_pitch: float | None = None) -> None:
        super().__init__(message)
        self.suggested_pitch = suggested_pitch


from .result import Abnormality


class PurePythonResult(NamedTuple):
    sa_um: float
    sq_um: float
    svr_um: float
    processed_points: int
    variogram_bins_um: np.ndarray
    variogram_counts: np.ndarray
    grid_width: int
    grid_height: int
    grid_origin_mm: np.ndarray
    grid_z_mm: np.ndarray
    elevation_grid_mm: np.ndarray = None
    noise_floor_um: float = 0.0
    svr_raw_um: float = 0.0
    plane_centroid_mm: np.ndarray = None
    plane_normal: np.ndarray = None
    plane_x_axis: np.ndarray = None
    plane_y_axis: np.ndarray = None
    abnormalities: tuple[Abnormality, ...] = ()
    svr_clean_um: float | None = None
    abnormal_mask: np.ndarray = None
    heatmap_grid_um: np.ndarray = None


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
    """Align point cloud surface normal to +Z axis and oriented bounding box (OBB) to X/Y axes."""
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

    # 2D Oriented Bounding Box (OBB) search to align X/Y axes with physical edges
    if len(centered) > 20:
        u0 = centered @ eig_vecs[:, 0]
        v0 = centered @ eig_vecs[:, 1]
        try:
            hull = ConvexHull(np.column_stack([u0, v0]))
            uh = u0[hull.vertices]
            vh = v0[hull.vertices]
        except Exception:
            step = max(1, len(u0) // 5000)
            uh, vh = u0[::step], v0[::step]

        angles = np.linspace(0, np.pi / 2, 90)
        cos_a = np.cos(angles)[:, None]
        sin_a = np.sin(angles)[:, None]
        ru = cos_a * uh[None, :] - sin_a * vh[None, :]
        rv = sin_a * uh[None, :] + cos_a * vh[None, :]
        areas = np.ptp(ru, axis=1) * np.ptp(rv, axis=1)
        best_ang = float(angles[np.argmin(areas)])

        cos_b, sin_b = np.cos(best_ang), np.sin(best_ang)
        x_ax = cos_b * eig_vecs[:, 0] - sin_b * eig_vecs[:, 1]
        y_ax = sin_b * eig_vecs[:, 0] + cos_b * eig_vecs[:, 1]
        z_ax = eig_vecs[:, 2]
        eig_vecs = np.column_stack([x_ax, y_ax, z_ax])

    # Rotate points into plane coordinates (col 0 -> X, col 1 -> Y, col 2 -> Z)
    rotated = centered @ eig_vecs

    # Zero-center Z
    z_mean = rotated[:, 2].mean()
    rotated[:, 2] -= z_mean

    return rotated, centroid, eig_vecs


def rasterize_elevation_grid(
    rotated_pts_m: np.ndarray,
    pitch_m: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Rasterize 3D points onto regular 2D elevation grid using bilinear scatter interpolation.

    Preserves the true physical contour and organic boundaries of all scanned points.
    """
    min_x, min_y = float(rotated_pts_m[:, 0].min()), float(rotated_pts_m[:, 1].min())
    max_x, max_y = float(rotated_pts_m[:, 0].max()), float(rotated_pts_m[:, 1].max())

    steps_x = int(math.ceil((max_x - min_x) / pitch_m)) + 1
    steps_y = int(math.ceil((max_y - min_y) / pitch_m)) + 1
    pitch_mm = pitch_m * 1000.0

    if steps_x < 2 or steps_y < 2:
        raise ValueError(f"Insufficient surface span to form a 2D grid at pitch {pitch_mm:.2f} mm.")

    inv_pitch = 1.0 / pitch_m
    ux = (rotated_pts_m[:, 0] - min_x) * inv_pitch
    uy = (rotated_pts_m[:, 1] - min_y) * inv_pitch

    ix0 = np.clip(np.floor(ux).astype(np.int32), 0, max(0, steps_x - 2))
    iy0 = np.clip(np.floor(uy).astype(np.int32), 0, max(0, steps_y - 2))
    fx = np.clip(ux - ix0, 0.0, 1.0)
    fy = np.clip(uy - iy0, 0.0, 1.0)

    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy

    M = steps_y * steps_x
    z = rotated_pts_m[:, 2]
    idx00 = iy0 * steps_x + ix0
    idx10 = iy0 * steps_x + (ix0 + 1)
    idx01 = (iy0 + 1) * steps_x + ix0
    idx11 = (iy0 + 1) * steps_x + (ix0 + 1)

    grid_sum_w = (
        np.bincount(idx00, weights=w00, minlength=M)
        + np.bincount(idx10, weights=w10, minlength=M)
        + np.bincount(idx01, weights=w01, minlength=M)
        + np.bincount(idx11, weights=w11, minlength=M)
    ).reshape(steps_y, steps_x)

    grid_sum_zw = (
        np.bincount(idx00, weights=w00 * z, minlength=M)
        + np.bincount(idx10, weights=w10 * z, minlength=M)
        + np.bincount(idx01, weights=w01 * z, minlength=M)
        + np.bincount(idx11, weights=w11 * z, minlength=M)
    ).reshape(steps_y, steps_x)

    valid = grid_sum_w > 1e-5
    grid_z = np.zeros((steps_y, steps_x), dtype=np.float64)
    grid_z[valid] = grid_sum_zw[valid] / grid_sum_w[valid]
    grid_z[~valid] = np.nan

    span_x_mm = (max_x - min_x) * 1000.0
    span_y_mm = (max_y - min_y) * 1000.0
    box_area_mm2 = max(span_x_mm * span_y_mm, 1e-4)
    n_points = max(len(rotated_pts_m), 1)
    approx_spacing_mm = math.sqrt(box_area_mm2 / n_points)

    if approx_spacing_mm > pitch_mm * 2.0:
        # Suggest a pitch matching approximately 0.55x to 1.0x the point spacing (rounded up to nearest 0.1mm)
        suggested_pitch = max(0.40, math.ceil(round(approx_spacing_mm * 0.55, 2) * 10.0) / 10.0)
        if suggested_pitch <= pitch_mm:
            suggested_pitch = round(pitch_mm * 2.0, 1)
        raise SparseScanError(
            f"Insufficient contiguous surface area at grid_mm={pitch_mm:.2f} mm. "
            f"Average point spacing for this scan is ~{approx_spacing_mm:.2f} mm (~{n_points / box_area_mm2:.1f} pts/mm^2), "
            f"which is significantly coarser than the requested {pitch_mm:.2f} mm grid pitch (causing most raster cells to be empty). "
            f"Re-run with a coarser pitch matching your scan density: --grid-mm {suggested_pitch:.1f}",
            suggested_pitch=suggested_pitch,
        )

    # Trim only purely empty boundary rows and columns (preserving full organic shape)
    valid_rows = np.where(valid.any(axis=1))[0]
    valid_cols = np.where(valid.any(axis=0))[0]
    if len(valid_rows) == 0 or len(valid_cols) == 0:
        raise ValueError(
            f"Insufficient contiguous surface area to form a valid roughness grid at grid_mm={pitch_mm:.2f} mm."
        )
    y_begin, y_end = int(valid_rows[0]), int(valid_rows[-1] + 1)
    x_begin, x_end = int(valid_cols[0]), int(valid_cols[-1] + 1)
    grid_z = grid_z[y_begin:y_end, x_begin:x_end]
    raw_valid = valid[y_begin:y_end, x_begin:x_end].copy()
    origin_x = min_x + x_begin * pitch_m
    origin_y = min_y + y_begin * pitch_m

    filled_z = _fill_grid_holes(grid_z, raw_valid)
    return filled_z, raw_valid, origin_x, origin_y


def _fill_grid_holes(grid_z: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Fill unmeasured holes iteratively for Gaussian filtration stability."""
    filled_z = grid_z.copy()
    if not np.any(~valid_mask) or not np.any(valid_mask):
        return filled_z

    current_valid = valid_mask.copy()
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
        mean_z = np.nanmean(filled_z[valid_mask])
        filled_z[~current_valid] = mean_z if np.isfinite(mean_z) else 0.0

    return filled_z


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
    return ndimage.convolve1d(arr, k_1d, axis=axis, mode="constant", cval=0.0)


def _normalized_convolve_2d(arr: np.ndarray, k_1d: np.ndarray) -> np.ndarray:
    """Separable 2D convolution with boundary partial-kernel normalization matching cloud_viewer.cpp."""
    # Pass 1: horizontal (X) - 1D normalization denominator broadcasted across rows
    den_x = _convolve1d_constant(np.ones(arr.shape[1], dtype=np.float64), k_1d, axis=0)[None, :]
    num_x = _convolve1d_constant(arr, k_1d, axis=1)
    step1 = num_x / np.maximum(den_x, 1e-12)
    # Pass 2: vertical (Y) - 1D normalization denominator broadcasted across columns
    den_y = _convolve1d_constant(np.ones(arr.shape[0], dtype=np.float64), k_1d, axis=0)[:, None]
    num_y = _convolve1d_constant(step1, k_1d, axis=0)
    return num_y / np.maximum(den_y, 1e-12)


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
    fast_shape = (next_fast_len(shape[0]), next_fast_len(shape[1]))
    sp1 = np.fft.rfft2(in1, s=fast_shape)
    sp2 = np.fft.rfft2(in2, s=fast_shape)
    ret = np.fft.irfft2(sp1 * sp2, s=fast_shape)
    f0 = (s2[0] - 1) // 2
    f1 = (s2[1] - 1) // 2
    return ret[f0 : f0 + s1[0], f1 : f1 + s1[1]]


def compute_variogram_and_svr(
    roughness_z_m: np.ndarray,
    pitch_m: float,
    points_on_var: int = 10,
    span_m: float = 0.0005,
    valid_mask: np.ndarray | None = None,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Compute ASTM WK92969 Sa, Sq, Svr, and variogram buckets.

    Uses 2D FFT autocorrelation for O(1) pairwise displacement accumulation (~38 ms).
    """
    height, width = roughness_z_m.shape

    # Only measured, finite cells can form valid variogram pairs.
    mask = np.isfinite(roughness_z_m)
    if valid_mask is not None:
        mask &= valid_mask
    if not np.any(mask):
        raise ValueError("No measured cells available for variogram calculation")
    valid_z = roughness_z_m[mask]
    sa_um = float(np.mean(np.abs(valid_z)) * 1e6)
    sq_um = float(np.sqrt(np.mean(valid_z**2)) * 1e6)

    max_ev_m = points_on_var * span_m
    max_cells = int(math.ceil(max_ev_m / pitch_m))
    z_um = np.where(mask, roughness_z_m, 0.0) * 1e6
    weights = mask.astype(np.float64)
    pad_h = height + max_cells + 1
    pad_w = width + max_cells + 1

    # Cross-correlations count measured pairs and their two squared terms.
    z_fft = np.fft.rfft2(z_um, s=(pad_h, pad_w))
    mask_fft = np.fft.rfft2(weights, s=(pad_h, pad_w))
    z2_fft = np.fft.rfft2(z_um**2, s=(pad_h, pad_w))
    autocorr = np.fft.irfft2(z_fft * np.conj(z_fft), s=(pad_h, pad_w))
    valid_pairs = np.fft.irfft2(mask_fft * np.conj(mask_fft), s=(pad_h, pad_w))
    src_sq = np.fft.irfft2(z2_fft * np.conj(mask_fft), s=(pad_h, pad_w))
    dst_sq = np.fft.irfft2(mask_fft * np.conj(z2_fft), s=(pad_h, pad_w))

    var_sums = np.zeros(points_on_var, dtype=np.float64)
    var_counts = np.zeros(points_on_var, dtype=np.int64)
    for dy in range(max_cells + 1):
        for dx in range(-max_cells, max_cells + 1):
            if dy == 0 and dx <= 0:
                continue
            dist = math.hypot(dx * pitch_m, dy * pitch_m)
            b = int(math.floor(dist / span_m))
            if b >= points_on_var:
                continue
            col = dx if dx >= 0 else pad_w + dx
            count = int(round(float(valid_pairs[dy, col])))
            if count <= 0:
                continue
            diff_sq = max(0.0, float(src_sq[dy, col] + dst_sq[dy, col] - 2.0 * autocorr[dy, col]))
            var_sums[b] += 2.0 * diff_sq
            var_counts[b] += 2 * count

    var_bins_um = np.zeros(points_on_var, dtype=np.float64)
    occupied = var_counts > 0
    var_bins_um[occupied] = np.sqrt(np.maximum(0.0, var_sums[occupied] / (2.0 * var_counts[occupied])))
    # WK92969 §9.2 defines N(b) as the number of configured distance buckets.
    svr_um = float(np.sum(var_bins_um) / points_on_var)

    return sa_um, sq_um, svr_um, var_bins_um, var_counts


def compute_heatmap_grid(
    roughness_grid_mm: np.ndarray,
    pitch_mm: float,
    radius_mm: float = 5.0,
    min_support_cells: int = 12,
    disc_averaging: bool | None = None,
) -> np.ndarray:
    """Compute local Svr spatial roughness heatmap (in µm) via 2D FFT convolution.

    Args:
        roughness_grid_mm: 2D array of Gaussian-filtered roughness elevation in mm.
        pitch_mm: Grid cell spacing in mm.
        radius_mm: Local circular evaluation radius in mm (default: 5.0 mm).
        min_support_cells: Minimum number of valid neighboring cells within radius_mm
            required to evaluate local variance (default: 12).
        disc_averaging: [Legacy] If True, compute legacy disc-averaged RMS pair variation:
            sqrt(2 * Var(z)). If False (default / ASTM WK92969 §9.2 standard), compute
            point-centered semivariance: sqrt(0.5 * [(z_i - mean_local)^2 + Var(z)]),
            preserving sharp local features (peaks and pits). If None, checks the
            'SVR_DISC_AVERAGING' environment variable (defaulting to False).

    Returns:
        2D float64 array of local Svr values in micrometers (µm).
    """
    if roughness_grid_mm.ndim != 2:
        raise ValueError("Expected a 2D roughness elevation grid")
    if pitch_mm <= 0 or radius_mm <= 0:
        raise ValueError("pitch_mm and radius_mm must be positive")

    if disc_averaging is None:
        disc_averaging = os.environ.get("SVR_DISC_AVERAGING", "").strip().lower() in ("1", "true", "yes", "on")

    max_cells = int(math.ceil(radius_mm / pitch_mm))
    z_um = roughness_grid_mm * 1000.0

    y_k, x_k = np.ogrid[-max_cells : max_cells + 1, -max_cells : max_cells + 1]
    dist_k = np.sqrt((x_k * pitch_mm) ** 2 + (y_k * pitch_mm) ** 2)
    kernel = ((dist_k <= radius_mm) & (dist_k > 0)).astype(np.float64)

    finite_mask = np.isfinite(z_um)
    z_clean = np.where(finite_mask, z_um, 0.0)
    w_valid = finite_mask.astype(np.float64)

    s1 = z_clean.shape
    s2 = kernel.shape
    shape = (s1[0] + s2[0] - 1, s1[1] + s2[1] - 1)
    fast_shape = (next_fast_len(shape[0]), next_fast_len(shape[1]))
    sp_k = np.fft.rfft2(kernel, s=fast_shape)
    f0 = (s2[0] - 1) // 2
    f1 = (s2[1] - 1) // 2

    def _conv_k(arr: np.ndarray) -> np.ndarray:
        sp_a = np.fft.rfft2(arr, s=fast_shape)
        ret = np.fft.irfft2(sp_a * sp_k, s=fast_shape)
        return ret[f0 : f0 + s1[0], f1 : f1 + s1[1]]

    w_local = _conv_k(w_valid)
    conv_z = _conv_k(z_clean)
    conv_z2 = _conv_k(z_clean**2)

    # Detrended local neighborhood statistics:
    mean_local = conv_z / np.maximum(w_local, 1e-12)
    var_local = np.maximum(0.0, (conv_z2 / np.maximum(w_local, 1e-12)) - mean_local**2)

    min_support = max(4.0, float(min_support_cells))
    valid_cells = (w_local >= (min_support - 1e-6)) & finite_mask
    heatmap_um = np.full_like(z_um, np.nan)

    if disc_averaging:
        # Disc-averaged RMS pair variation:
        # In ASTM WK92969 §9.2, v(d) = sqrt(1/(2N) sum (zi - zj)^2).
        # The expected RMS pair difference across the circular neighborhood is sqrt(2 * Var(z)).
        heatmap_um[valid_cells] = np.sqrt(2.0 * var_local[valid_cells])
    else:
        # Point-centered semivariance matching ASTM WK92969 §9.2:
        # gamma(i) = 1/(2N) * sum_j (z_i - z_j)^2 = 0.5 * [ (z_i - mean_local)^2 + var_local ]
        # Preserves needle-sharp local point contrast (individual grain peaks and pores)
        # while evaluating height differences across the full evaluation radius_mm.
        msd_point = (z_clean - mean_local) ** 2 + var_local
        local_gamma = 0.5 * msd_point
        heatmap_um[valid_cells] = np.sqrt(np.maximum(0.0, local_gamma[valid_cells]))

    return heatmap_um


def interpolate_heatmap_at_points(
    hmap: np.ndarray,
    origin_x_mm: float,
    origin_y_mm: float,
    pitch_mm: float,
    u_mm: np.ndarray,
    v_mm: np.ndarray,
) -> np.ndarray:
    """Continuously evaluate local Svr heatmap at arbitrary point coordinates via bilinear interpolation.

    Performs smooth, mathematically rigorous sampling of the areal roughness field
    without nearest-neighbor inpainting or discrete grid index clipping.
    """
    fx = (u_mm - origin_x_mm) / pitch_mm
    fy = (v_mm - origin_y_mm) / pitch_mm

    h, w = hmap.shape
    if h < 2 or w < 2 or len(u_mm) == 0:
        return np.full(len(u_mm), np.nan, dtype=np.float32)

    ix0 = np.clip(np.floor(fx).astype(np.int32), 0, max(0, w - 2))
    iy0 = np.clip(np.floor(fy).astype(np.int32), 0, max(0, h - 2))
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    dx = np.clip(fx - ix0, 0.0, 1.0)
    dy = np.clip(fy - iy0, 0.0, 1.0)

    v00 = hmap[iy0, ix0]
    v10 = hmap[iy0, ix1]
    v01 = hmap[iy1, ix0]
    v11 = hmap[iy1, ix1]

    w00 = (1.0 - dx) * (1.0 - dy)
    w10 = dx * (1.0 - dy)
    w01 = (1.0 - dx) * dy
    w11 = dx * dy

    m00 = np.isfinite(v00)
    m10 = np.isfinite(v10)
    m01 = np.isfinite(v01)
    m11 = np.isfinite(v11)

    sum_vals = (
        np.where(m00, w00 * v00, 0.0)
        + np.where(m10, w10 * v10, 0.0)
        + np.where(m01, w01 * v01, 0.0)
        + np.where(m11, w11 * v11, 0.0)
    )
    sum_weights = (
        np.where(m00, w00, 0.0)
        + np.where(m10, w10, 0.0)
        + np.where(m01, w01, 0.0)
        + np.where(m11, w11, 0.0)
    )

    out = np.where(sum_weights > 1e-6, sum_vals / np.maximum(sum_weights, 1e-12), np.nan)
    return out.astype(np.float32)


def detect_surface_abnormalities(
    roughness_grid_mm: np.ndarray,
    pitch_mm: float,
    global_svr_um: float,
    threshold_factor: float = 2.0,
    min_area_mm2: float = 2.0,
    valid_mask: np.ndarray | None = None,
    heatmap_grid_um: np.ndarray | None = None,
) -> tuple[list[Abnormality], np.ndarray]:
    """Identify and segment surface abnormalities per ASTM WK92969 §8.7.

    Points are defined as abnormalities if their local roughness parameter (local S_VR)
    is >= threshold_factor * global_svr_um (default 2.0x) AND they form a connected
    region with area >= min_area_mm2 (default 2.0 mm²).

    Perimeter edge artifacts within 2 grid cells of the valid patch border are suppressed
    via binary erosion to prevent false boundary triggers.

    Returns:
        tuple of (list of Abnormality objects, 2D boolean mask of abnormal cells).
    """
    if roughness_grid_mm.ndim != 2:
        raise ValueError("Expected a 2D roughness elevation grid")

    if valid_mask is None:
        valid_mask = np.isfinite(roughness_grid_mm)
    else:
        valid_mask = valid_mask & np.isfinite(roughness_grid_mm)

    if not np.any(valid_mask) or global_svr_um <= 0.0:
        return [], np.zeros_like(valid_mask, dtype=bool)

    if heatmap_grid_um is None:
        heatmap_grid_um = compute_heatmap_grid(roughness_grid_mm, pitch_mm)

    threshold_um = float(threshold_factor * global_svr_um)
    candidates = valid_mask & np.isfinite(heatmap_grid_um) & (heatmap_grid_um >= threshold_um)

    # Suppress perimeter edge artifacts (2 * pitch_mm margin via erosion)
    if np.any(valid_mask):
        eroded_boundary = ndimage.binary_erosion(valid_mask, iterations=2)
        candidates &= eroded_boundary

    if not np.any(candidates):
        return [], np.zeros_like(valid_mask, dtype=bool)

    structure = np.ones((3, 3), dtype=int)
    labeled, num_features = ndimage.label(candidates, structure=structure)
    if num_features == 0:
        return [], np.zeros_like(valid_mask, dtype=bool)

    cell_area_mm2 = pitch_mm * pitch_mm
    min_cells = max(1, int(math.ceil(min_area_mm2 / cell_area_mm2)))
    cluster_sizes = np.bincount(labeled.ravel())

    abnormalities: list[Abnormality] = []
    abnormal_mask = np.zeros_like(valid_mask, dtype=bool)

    cluster_counter = 1
    for feat_id in range(1, num_features + 1):
        c_count = int(cluster_sizes[feat_id]) if feat_id < len(cluster_sizes) else 0
        if c_count < min_cells:
            continue

        c_mask = (labeled == feat_id)
        abnormal_mask |= c_mask

        area_mm2 = float(c_count * cell_area_mm2)
        diameter_mm = float(2.0 * math.sqrt(area_mm2 / math.pi))

        cluster_z_vals = roughness_grid_mm[c_mask]
        mean_z = float(np.mean(cluster_z_vals))
        kind = "protrusion" if mean_z >= 0 else "depression"
        peak_dev_um = float(np.max(np.abs(cluster_z_vals)) * 1000.0)

        y_indices, x_indices = np.where(c_mask)
        centroid_u = float(np.mean(x_indices) * pitch_mm)
        centroid_v = float(np.mean(y_indices) * pitch_mm)

        abnormalities.append(
            Abnormality(
                id=cluster_counter,
                kind=kind,
                area_mm2=round(area_mm2, 4),
                diameter_mm=round(diameter_mm, 4),
                peak_deviation_um=round(peak_dev_um, 2),
                centroid_uv_mm=(round(centroid_u, 3), round(centroid_v, 3)),
                point_count=c_count,
            )
        )
        cluster_counter += 1

    return abnormalities, abnormal_mask


def analyze_pure_python(
    points_xyz_mm: np.ndarray,
    voxel_size_mm: float = 0.20,
    short_cutoff_mm: float = 1.0,
    long_cutoff_mm: float = 25.0,
    variogram_points: int = 10,
    variogram_span_mm: float = 0.5,
    subtract_noise: bool = True,
    gaussian_mesh: bool = True,
    detect_abnormalities: bool = False,
    abnormality_threshold: float = 2.0,
    min_abnormality_area_mm2: float = 2.0,
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

    # Step 2: Voxel Grid Downsampling (fast robust plane fitting)
    downsampled_m = voxel_downsample(pts_m, voxel_size_m)

    # Step 3: Form Removal via PCA Plane Fitting & 2D OBB Alignment
    _, _, eig_vecs = pca_align_plane(downsampled_m)
    all_rotated_m = pts_m @ eig_vecs

    # Step 4: 2.5D Regular Elevation Grid Rasterization (encompassing all scanned points)
    grid_z_m, valid_mask, origin_x_m, origin_y_m = rasterize_elevation_grid(all_rotated_m, voxel_size_m)

    # Dynamic instrument noise estimation on elevation grid via 2D discrete Laplacian MAD:
    lap = _fft_convolve2d_same(grid_z_m, np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64))
    valid_lap = valid_mask & np.isfinite(lap)
    if np.any(valid_lap):
        mad = float(np.median(np.abs(lap[valid_lap] - np.median(lap[valid_lap]))))
        noise_floor_um = float((mad / 0.6745 / math.sqrt(20.0)) * 1e6)
    else:
        noise_floor_um = 0.0

    # Step 5: Dual-Pass ISO 16610-61 Gaussian Filtration
    if gaussian_mesh:
        roughness_z_m = apply_dual_pass_gaussian_filter(grid_z_m, voxel_size_m, short_cutoff_m, long_cutoff_m)
    else:
        roughness_z_m = grid_z_m

    # Step 6: Variogram & Svr Metrology on valid points
    sa_um, sq_um, svr_raw_um, var_bins_um, var_counts = compute_variogram_and_svr(
        roughness_z_m,
        voxel_size_m,
        variogram_points,
        span_m,
        valid_mask=valid_mask,
    )

    # Single definitive Svr measurement (with dynamic noise floor subtraction if enabled)
    if subtract_noise and noise_floor_um > 0.0:
        svr_um = float(math.sqrt(max(0.0, svr_raw_um**2 - noise_floor_um**2)))
    else:
        svr_um = svr_raw_um

    height, width = roughness_z_m.shape
    grid_z_mm = np.where(valid_mask, roughness_z_m * 1000.0, np.nan)
    elevation_grid_mm = np.where(valid_mask, grid_z_m * 1000.0, np.nan)

    # Step 7: Abnormality Detection & Recalculation (ASTM WK92969 §8.7)
    abnormalities: tuple[Abnormality, ...] = ()
    svr_clean_um: float | None = None
    abnormal_mask: np.ndarray | None = None
    heatmap_grid_um: np.ndarray | None = None

    if detect_abnormalities:
        heatmap_grid_um = compute_heatmap_grid(grid_z_mm, voxel_size_mm)
        abn_list, abn_mask = detect_surface_abnormalities(
            grid_z_mm,
            voxel_size_mm,
            svr_um,
            threshold_factor=abnormality_threshold,
            min_area_mm2=min_abnormality_area_mm2,
            valid_mask=valid_mask,
            heatmap_grid_um=heatmap_grid_um,
        )
        abnormalities = tuple(abn_list)
        abnormal_mask = abn_mask
        if abnormalities:
            clean_valid_mask = valid_mask & ~abnormal_mask
            if np.any(clean_valid_mask):
                clean_grid_z_m = _fill_grid_holes(grid_z_m, clean_valid_mask)
                if gaussian_mesh:
                    clean_roughness_z_m = apply_dual_pass_gaussian_filter(clean_grid_z_m, voxel_size_m, short_cutoff_m, long_cutoff_m)
                else:
                    clean_roughness_z_m = clean_grid_z_m

                _, _, clean_svr_raw_um, _, _ = compute_variogram_and_svr(
                    clean_roughness_z_m,
                    voxel_size_m,
                    variogram_points,
                    span_m,
                    valid_mask=clean_valid_mask,
                )
                if subtract_noise and noise_floor_um > 0.0:
                    svr_clean_um = float(math.sqrt(max(0.0, clean_svr_raw_um**2 - noise_floor_um**2)))
                else:
                    svr_clean_um = clean_svr_raw_um
            else:
                svr_clean_um = svr_um
        else:
            svr_clean_um = svr_um

    return PurePythonResult(
        sa_um=sa_um,
        sq_um=sq_um,
        svr_um=svr_um,
        processed_points=len(downsampled_m),
        variogram_bins_um=var_bins_um,
        variogram_counts=var_counts,
        grid_width=width,
        grid_height=height,
        grid_origin_mm=np.array([origin_x_m * 1000.0, origin_y_m * 1000.0, voxel_size_mm], dtype=np.float64),
        grid_z_mm=grid_z_mm,
        elevation_grid_mm=elevation_grid_mm,
        noise_floor_um=noise_floor_um,
        svr_raw_um=svr_raw_um,
        plane_centroid_mm=centroid,
        plane_normal=eig_vecs[:, 2],
        plane_x_axis=eig_vecs[:, 0],
        plane_y_axis=eig_vecs[:, 1],
        abnormalities=abnormalities,
        svr_clean_um=svr_clean_um,
        abnormal_mask=abnormal_mask,
        heatmap_grid_um=heatmap_grid_um,
    )

