from __future__ import annotations

import argparse
from pathlib import Path

from .analyze import analyze_file
from .config import RoughnessConfig
from .result import format_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate surface roughness from a point-cloud file.")
    parser.add_argument("input", type=Path, help="Input PLY, PCD, STL, OBJ, CSV, XYZ, TXT, TSV, NPY, or NPZ file")
    parser.add_argument("--grid-mm", type=float, default=0.30)
    parser.add_argument("--short-cutoff-mm", type=float, default=0.0)
    parser.add_argument("--long-cutoff-mm", type=float, default=0.0)
    parser.add_argument("--gaussian-mesh", action="store_true",
                        help="Use the recovered Cloud-Viewer legacy Gaussian mesh path")
    parser.add_argument("--metrics-out", type=Path)
    parser.add_argument("--grid-out", type=Path)
    args = parser.parse_args()

    result = analyze_file(
        args.input,
        RoughnessConfig(
            grid_mm=args.grid_mm,
            short_cutoff_mm=args.short_cutoff_mm,
            long_cutoff_mm=args.long_cutoff_mm,
            gaussian_mesh=args.gaussian_mesh,
        ),
    )
    print(format_report(result))
    if args.metrics_out:
        result.save_metrics_json(args.metrics_out)
    if args.grid_out:
        result.save_grid_npz(args.grid_out)


if __name__ == "__main__":
    main()
