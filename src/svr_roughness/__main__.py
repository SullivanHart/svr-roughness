from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analyze import analyze_file, analyze_object
from .config import RoughnessConfig
from .decomposition import DecompositionConfig
from .result import format_report


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Calculate surface roughness from a point-cloud file.")
    parser.add_argument("input", type=Path, help="Input PLY, PCD, STL, OBJ, CSV, XYZ, TXT, TSV, NPY, or NPZ file")
    parser.add_argument(
        "--3d",
        "--decompose",
        dest="decompose_3d",
        action="store_true",
        help="Decompose 3D object scan into planar faces and evaluate ASTM roughness per face",
    )
    parser.add_argument("--grid-mm", type=float, default=0.20, help="Grid pitch / downsample in mm (default: 0.20)")
    parser.add_argument("--short-cutoff-mm", type=float, default=1.0, help="Short cutoff lambda_s in mm (default: 1.0)")
    parser.add_argument("--long-cutoff-mm", type=float, default=25.0, help="Long cutoff lambda_c in mm (default: 25.0)")
    parser.add_argument(
        "--gaussian-mesh", action="store_true", default=True, help="Enforce ISO 16610-61 Gaussian filtration"
    )
    parser.add_argument(
        "--edge-margin-mm",
        type=float,
        default=1.5,
        help="Edge margin buffer in mm to remove rounded fillets and parting lines (default: 1.5)",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=12,
        help="Maximum number of planar faces to extract in 3D mode (default: 12)",
    )
    parser.add_argument("--metrics-out", type=Path, help="Path to save metrics as JSON")
    parser.add_argument("--grid-out", type=Path, help="Path to save 2D roughness grid as NPZ")
    parser.add_argument("--heatmap-out", type=Path, help="Path to save 2D Svr heatmap as NPY or NPZ")
    args = parser.parse_args()

    config = RoughnessConfig(
        grid_mm=args.grid_mm,
        short_cutoff_mm=args.short_cutoff_mm,
        long_cutoff_mm=args.long_cutoff_mm,
        gaussian_mesh=args.gaussian_mesh,
    )

    if args.decompose_3d:
        decomp_cfg = DecompositionConfig(
            edge_margin_mm=args.edge_margin_mm,
            max_faces=args.max_faces,
        )
        obj_result = analyze_object(args.input, config=config, decomp_config=decomp_cfg)
        print(obj_result.format_report())
        if args.metrics_out:
            args.metrics_out.write_text(json.dumps(obj_result.to_dict(), indent=2), encoding="utf-8")
        if args.grid_out:
            # Save grids from all detected patches into a single NPZ
            import numpy as np

            grids = {f"face_{p.patch_id}": p.roughness.grid for p in obj_result.patches if p.roughness.grid is not None}
            np.savez(args.grid_out, **grids)
        return

    result = analyze_file(args.input, config)
    print(format_report(result))
    if args.metrics_out:
        result.save_metrics_json(args.metrics_out)
    if args.grid_out:
        result.save_grid_npz(args.grid_out)
    if args.heatmap_out:
        hmap = result.heatmap()
        if args.heatmap_out.suffix == ".npz":
            import numpy as np

            np.savez(args.heatmap_out, heatmap=hmap, pitch_mm=result.grid_pitch_mm, origin=result.grid_origin_mm)
        else:
            import numpy as np

            np.save(args.heatmap_out, hmap)


if __name__ == "__main__":
    main()

