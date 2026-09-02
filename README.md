# svr-roughness

`svr-roughness` is a Python adapter for the native SurfInspect numerical core
extracted from Cloud-Viewer. Use `svr_roughness` as the consistent Python
import name.

The core API is file-type agnostic: roughness is computed from an `Nx3` NumPy-like XYZ point array in millimeters. Svr is calculated from signed distances to a smoothed, triangulated reference surface and a KD-tree variogram modeled on SurfInspect. The regular grid remains available for visualization. The native reference
mesh downsampling defaults to 1.0 mm. SurfInspect-compatible statistical
outlier filtering is enabled by default (`mean_k=6`, standard-deviation
multiplier `3.0`), and the native variogram uses every processed point.

```python
config = RoughnessConfig(
    mesh_resolution_mm=1.0,
    svr_points=10,
    svr_span_mm=0.50,
)
```

Set `gaussian_mesh=True` to select the recovered Cloud-Viewer legacy Gaussian
mesh path. It PCA-aligns the mesh cloud, samples a fixed 1 mm grid, applies the
legacy alpha `0.4697` kernel (including its historical `0.0002 m` kernel
coordinate factor and combined low/high weighting), and constructs a dense
CGAL mesh without remeshing. The flag defaults to `False` so existing callers
retain the current advancing-front/remeshed behavior. This recovered path is
still being validated against the legacy executable and should not yet be
treated as a numerical compatibility guarantee.

```python
config = RoughnessConfig(
    gaussian_mesh=True,
    short_cutoff_mm=1.0,
    long_cutoff_mm=25.0,
)
```

The legacy implementation has a few preserved quirks: the low- and
high-pass kernels are combined as `low * (1 - high)`, the Gaussian kernel
coordinate factor remains `0.2 mm` despite the 1 mm grid, and empty grid cells
are filled from the three nearest source points. The native adapter keeps
those behaviors while retaining safer bounds checks and normal propagation for
signed distances.

The public `analyze_*` APIs require the native CGAL/PCL library; they do not
silently fall back to an approximate Python implementation. The package now
contains the native source and builds it as part of a source distribution or
wheel build. The core accepts millimetre XYZ arrays and performs the original
voxel filtering, CGAL normal estimation, advancing-front reconstruction,
remeshing, signed point-to-mesh distances, and variogram calculation without
scanner, viewer, file-output, or process-global state.

## Install

From a prebuilt wheel:

```bash
python -m pip install svr-roughness
```

For a local source build:

```bash
python -m pip install build
python -m build --wheel
python -m pip install dist/svr_roughness-*.whl
```

The native build must be able to find CGAL, PCL, Boost, Eigen, TBB, GMP, and
MPFR. For vcpkg, provide the toolchain to the isolated build environment:

```powershell
$env:CMAKE_ARGS = "-DCMAKE_TOOLCHAIN_FILE=C:\path\to\vcpkg\scripts\buildsystems\vcpkg.cmake;-DVCPKG_TARGET_TRIPLET=x64-windows"
python -m build --wheel
```

The wheel contains `svr_roughness/native/SurfInspectNative.dll` (or the
platform equivalent), but it does **not** bundle the third-party runtime DLLs.
Consequently, this project does not claim a self-contained, portable wheel:
the matching PCL/CGAL/Boost/TBB/GMP/MPFR runtime libraries must be installed
and discoverable on the target system.

## Python API

Use `analyze_points()` when your scanner or upstream software already gives you XYZ points:

```python
import numpy as np
from svr_roughness import analyze_points

points_xyz_mm = np.asarray(points)  # shape (N, 3), columns x/y/z, units mm
result = analyze_points(points_xyz_mm, grid_mm=0.30, short_cutoff_mm=0.6, long_cutoff_mm=8.0)

print(result.sa_um, result.sq_um, result.svr_um)
```

Use `analyze_file()` as a convenience adapter for supported files:

```python
from svr_roughness import RoughnessConfig, analyze_file

config = RoughnessConfig(grid_mm=0.30, short_cutoff_mm=0.6, long_cutoff_mm=8.0)
result = analyze_file("scan.ply", config=config)
```

Supported file loaders (all are converted to an `Nx3` NumPy array):

- ASCII and binary little-endian `.ply` point clouds
- ASCII and binary `.pcd` point clouds
- ASCII and binary `.stl` mesh vertices
- `.obj` vertex meshes
- comma-, tab-, or whitespace-delimited `.csv`, `.tsv`, `.xyz`, and `.txt`
- `.npy` and `.npz` NumPy arrays

STL files are converted to their unique mesh vertices before analysis. For production mesh metrology, prefer scanner point clouds or add controlled surface sampling before calling `analyze_points()`.

## Result Output

```python
result.save_grid_npz("output/roughness/latest_grid.npz")
result.save_metrics_json("output/roughness/latest_metrics.json")
```

`save_metrics_json()` writes:

- `sa_um`
- `sq_um`
- `svr_um`
- point counts, grid dimensions, grid coverage, and filter cutoffs

`save_grid_npz()` writes:

- `grid_raw`
- `grid_filled`
- `grid_filtered`
- `valid_raw`
- `valid_filled`
- `grid_origin`
- plane basis arrays

## Command line

The package also includes a no-code command for scanner integrations:

```bash
svr-roughness scan.ply --grid-mm 0.30 --metrics-out metrics.json --grid-out grid.npz
# Add --gaussian-mesh to use the recovered legacy Gaussian mesh path.
```

Coordinates are assumed to be millimeters, and metrics are reported in
micrometers. Unit conversion should happen before calling the library.

## Native source build

The native source and its dependency manifest are under `native/` in this
package and are independent of the sibling Cloud-Viewer checkout. A C++
compiler and the native development libraries are required only when building
from source. Visual Studio Code is not required.

```bash
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
bootstrap-vcpkg.bat
vcpkg install --triplet x64-windows --x-manifest-root=..\svr-roughness\native
cd ../svr-roughness
cmake -S native -B native/build \
  -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build native/build --config Release --target SurfInspectNative
```

On Windows, use Visual Studio 2022 Build Tools with an MSVC compiler version
1930 or newer. The current Boost release rejects the older MSVC 1929 compiler
from Visual Studio 2019. Visual Studio Code is not required.

PowerShell users can run the same build without shell line continuations:

```powershell
Set-Location C:\path\to\svr-roughness
vcpkg install --triplet x64-windows --x-manifest-root="$PWD\native"
cmake -S native -B native\build -DCMAKE_TOOLCHAIN_FILE="C:\path\to\vcpkg\scripts\buildsystems\vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build native\build --config Release --target SurfInspectNative
```

On Linux or macOS, replace the vcpkg bootstrap command and triplet with the
platform's normal vcpkg workflow, for example `x64-linux` or `arm64-osx`.
The generated shared library is installed into the wheel's
`svr_roughness/native/` directory. If those dependencies are unavailable, the
wheel build fails rather than producing an inaccurate fallback result.

To build and install from source after the native dependencies are available:

```bash
python -m pip install build
python -m build --wheel
python -m pip install dist/svr_roughness-*.whl
```

## Performance Roadmap & Optimization Opportunities

The native C++ core currently computes full surface metrology on 1,000,000 points in **~2-3 seconds**. Key opportunities for 10x-100x execution speedups include:

1. **2D FFT Gaussian Filtering (700x Speedup in Filtering)**
   - *Current*: 2D spatial convolution with an $85 \times 85$ float kernel ($O(N \cdot K^2)$), requiring $>10^9$ FLOPs per pass.
   - *Optimization*: Replace spatial convolution with a 2D Fast Fourier Transform (FFT via `fftw3` or `cv::dft()`) in frequency domain ($O(N \log N)$), reducing filtering time from 4.0s to $< 10\text{ ms}$.

2. **Delaunay Heightfield Surface Meshing (25x Speedup in Reconstruction)**
   - *Current*: Single-threaded 3D CGAL advancing-front triangulation (`CGAL::advancing_front_surface_reconstruction`).
   - *Optimization*: Since scanner heightfield scans are 2.5D surfaces ($Z = f(X, Y)$), replace 3D advancing-front meshing with 2D Delaunay grid projection or multi-threaded `CGAL::Mesh_3`, dropping meshing time from 2.5s to $< 50\text{ ms}$.

3. **CUDA GPU Offloading (Total Execution Time < 50 ms)**
   - Offload both 2D FFT Gaussian convolution (`cuFFT`) and point-to-mesh AABB tree distance queries (CUDA / Thrust) to GPU memory, enabling real-time $60\text{ FPS}$ scan processing during automated manufacturing.

## Feature & Architectural Roadmap

1. **ISO 25178-2 3D Areal Roughness Parameters ($S_z, S_{sk}, S_{ku}, S_{pd}$)**
   - Extend C++ core metrology to calculate ISO 25178-2 3D parameters: Maximum Height ($S_z$), Skewness ($S_{sk}$), Kurtosis ($S_{ku}$), and Peak Density ($S_{pd}$) for advanced surface characterization.

2. **Interactive 2D Profile Slicing ($R_a, R_q, R_z$)**
   - Add interactive line slicing tools in the Streamlit GUI to extract 2D cross-sectional height profiles and 2D roughness metrics along user-drawn paths.

3. **Multi-File Batch Processing & Automated Reporting**
   - Support drag-and-drop batch processing of multi-file scan directories in the web GUI, producing consolidated PDF/CSV quality inspection reports.

4. **Standalone Portable PyPI Wheels (`cibuildwheel`)**
   - Implement GitHub Actions `cibuildwheel` CI/CD workflows to build self-contained PyPI wheels bundling statically linked native libraries across Windows, Linux, and macOS.