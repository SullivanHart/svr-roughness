# svr-roughness

`svr-roughness` is a high-performance Python package for ASTM WK92969 and ISO 16610-61 areal surface roughness analysis ($S_a$, $S_q$, $S_{VR}$). It is developed by the Human-centered Advanced Manufacturing (HAM) Lab at Iowa State University.

The numerical core is implemented in pure Python using vectorized NumPy operations, eliminating all heavy external C++ compiler and DLL dependencies while providing 100% numerical fidelity to the reference SurfInspect metrology pipeline.

```python
import numpy as np
from svr_roughness import RoughnessConfig, analyze_points

# points shape (N, 3), units in millimeters
config = RoughnessConfig(
    grid_mm=0.20,
    short_cutoff_mm=1.0,
    long_cutoff_mm=25.0,
    svr_points=10,
    svr_span_mm=0.50,
)
result = analyze_points(points_xyz_mm, config=config)
print(f"Sa:  {result.sa_um:.3f} µm")
print(f"Sq:  {result.sq_um:.3f} µm")
print(f"Svr: {result.svr_um:.3f} µm")
```

## Features

- **Standard-Compliant Metrology**: Implements the full ASTM WK92969 pipeline:
  1. Centering and unit standardization
  2. $O(N)$ 3D voxel grid centroid downsampling
  3. PCA best-fit plane alignment
  4. 2.5D elevation grid rasterization with iterative hole filling and boundary cropping
  5. Dual-pass ISO 16610-61 Gaussian filtration ($\alpha = 0.4697$)
  6. 2D FFT autocorrelation variogram and local $S_{VR}$ dispersion map
- **Zero-Compiling Pure-Python**: Runs everywhere Python 3.10+ runs (Linux, macOS, Windows, Pyodide/WebAssembly) with only standard `numpy` and `scipy`.
- **Wide Scan Format Support**: Reads `.ply` (ASCII & binary), `.pcd` (ASCII & binary), `.stl` (ASCII & binary), `.obj`, delimited text (`.csv`, `.tsv`, `.xyz`, `.txt`), and NumPy arrays (`.npy`, `.npz`).
- **CLI and Web Integration**: Ships with a command-line interface (`svr-roughness`) and powers the client-side WebAssembly inspection dashboard.

## Install

Install from PyPI:

```bash
pip install svr-roughness
```

Or install from source:

```bash
pip install .
```

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

## Testing

Run the test suite using `pytest`:

```bash
pytest tests/ -v
```

The test suite covers:
- Complete I/O loading for all supported formats (ASCII/binary PLY, ASCII/binary STL, PCD, OBJ, CSV, XYZ, NPY/NPZ)
- $O(N)$ voxel downsampling and edge cases
- PCA plane alignment and coordinate frame invariants
- 2.5D elevation grid rasterization, boundary shaving, and iterative hole filling
- Dual-pass ISO 16610-61 Gaussian filtering ($\alpha = 0.4697$)
- Monotonicity, dispersion mapping, and Cauchy-Schwarz mathematical invariants ($S_q \ge S_a$)
- End-to-end ASTM WK92969 validation on SCRATA comparator standards

## Legacy C++ Native Core (Optional)

The historical C++ native bridge and CMake configuration are preserved under `native/` for reference and backwards compatibility. The standard pure-Python package no longer requires building or compiling native code, as the vectorized NumPy/SciPy engine provides identical numerical output with superior cross-platform portability.

## License

MIT License — Iowa State University HAM Lab.