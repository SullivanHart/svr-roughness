from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest


def test_cli_help():
    res = subprocess.run(
        [sys.executable, "-m", "svr_roughness", "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    assert "Calculate surface roughness from a point-cloud file" in res.stdout
    assert "--gaussian-mesh" in res.stdout
    assert "--no-gaussian-mesh" in res.stdout
    assert "--disc-averaging" in res.stdout


def test_cli_single_surface_run(tmp_path: Path):
    # Create simple planar CSV
    x, y = np.meshgrid(np.arange(0.0, 20.0, 0.5), np.arange(0.0, 20.0, 0.5))
    pts = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    csv_file = tmp_path / "plane.csv"
    np.savetxt(csv_file, pts, delimiter=",", header="x,y,z", comments="")

    metrics_out = tmp_path / "metrics.json"
    grid_out = tmp_path / "grid.npz"

    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "svr_roughness",
            str(csv_file),
            "--grid-mm",
            "0.5",
            "--metrics-out",
            str(metrics_out),
            "--grid-out",
            str(grid_out),
            "--no-gaussian-mesh",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    assert "Surface Variogram Roughness" in res.stdout
    assert metrics_out.is_file()
    assert grid_out.is_file()

    loaded = json.loads(metrics_out.read_text(encoding="utf-8"))
    assert "svr_um" in loaded
    assert "sa_um" in loaded


def test_cli_3d_mode(tmp_path: Path):
    # Create synthetic corner (2 planes) with >5000 points each
    x, y = np.meshgrid(np.arange(0.0, 75.0, 0.8), np.arange(0.0, 75.0, 0.8))
    p1 = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    p2 = np.column_stack((x.ravel(), np.zeros(x.size), y.ravel()))
    pts = np.vstack((p1, p2))
    csv_file = tmp_path / "corner.csv"
    np.savetxt(csv_file, pts, delimiter=",", header="x,y,z", comments="")

    metrics_out = tmp_path / "metrics_3d.json"

    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "svr_roughness",
            str(csv_file),
            "--3d",
            "--grid-mm",
            "0.5",
            "--metrics-out",
            str(metrics_out),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    assert "Multi-Surface 3D Object Roughness Report" in res.stdout
    assert metrics_out.is_file()
    loaded = json.loads(metrics_out.read_text(encoding="utf-8"))
    assert "patches" in loaded
