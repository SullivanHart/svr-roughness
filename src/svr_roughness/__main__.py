from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analyze import analyze_file
from .config import RoughnessConfig
from .result import format_report


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Calculate surface roughness from a point-cloud file.")
    parser.add_argument("input", type=Path, help="Input PLY, PCD, STL, OBJ, CSV, XYZ, TXT, TSV, NPY, or NPZ file")
    parser.add_argument("--grid-mm", type=float, default=0.20, help="Grid pitch / downsample in mm (default: 0.20)")
    parser.add_argument("--short-cutoff-mm", type=float, default=1.0, help="Short cutoff lambda_s in mm (default: 1.0)")
    parser.add_argument("--long-cutoff-mm", type=float, default=25.0, help="Long cutoff lambda_c in mm (default: 25.0)")
    parser.add_argument(
        "--gaussian-mesh", action="store_true", default=True, help="Enforce ISO 16610-61 Gaussian filtration"
    )
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

