from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from svr_roughness.config import RoughnessConfig
from svr_roughness.result import PlaneFit, RoughnessResult, format_report


def test_roughness_result_units_and_dimensions():
    grid = np.zeros((20, 30), dtype=np.float64)
    res = RoughnessResult(
        svr_um=25.4,
        sa_um=30.0,
        sq_um=40.0,
        points=6000,
        processed_points=5000,
        grid=grid,
        grid_pitch_mm=0.2,
        config=RoughnessConfig(),
    )
    # Unit conversions: 25.4 um = 0.0254 mm = 0.001 in
    assert pytest.approx(res.svr_mm, rel=1e-5) == 0.0254
    assert pytest.approx(res.svr_in, rel=1e-5) == 0.001

    # Dimensions without explicit PlaneFit coords (inferred from grid)
    assert pytest.approx(res.patch_width_mm) == 30 * 0.2  # 6.0 mm
    assert pytest.approx(res.patch_height_mm) == 20 * 0.2  # 4.0 mm
    assert pytest.approx(res.patch_area_mm2) == 24.0

    # Point density and spacing
    assert pytest.approx(res.point_density_pts_mm2) == 6000 / 24.0  # 250 pts/mm^2
    assert pytest.approx(res.average_point_spacing_mm, rel=1e-3) == 1.0 / np.sqrt(250)


def test_comparator_equivalents():
    # Test visual grade classification at various Svr thresholds
    grades_low = RoughnessResult(svr_um=10.0, sa_um=12.0, sq_um=15.0).comparator_equivalents()
    assert grades_low["SCRATA (A802)"] == "< A1"
    assert grades_low["GAR C-9"] == "C-9 200"
    assert grades_low["ACI SIS"] == "SIS-1"

    grades_high = RoughnessResult(svr_um=200.0, sa_um=220.0, sq_um=250.0).comparator_equivalents()
    assert grades_high["SCRATA (A802)"] == "> A4"
    assert grades_high["GAR C-9"] == "> C-9 900"
    assert grades_high["ACI SIS"] == "> SIS-4"


def test_to_dict_and_nan_handling():
    res = RoughnessResult(
        svr_um=float("nan"),
        sa_um=20.5,
        sq_um=float("inf"),
        points=100,
        processed_points=90,
    )
    d = res.to_dict()
    assert d["svr_um"] is None
    assert d["sa_um"] == 20.5
    assert d["sq_um"] is None
    assert d["points"] == 100
    # Must be JSON-serializable
    encoded = json.dumps(d)
    assert "20.5" in encoded


def test_format_report_pass_fail_and_warnings():
    res = RoughnessResult(
        svr_um=30.0,
        sa_um=25.0,
        sq_um=35.0,
        points=10000,
        processed_points=9000,
        grid=np.zeros((300, 300)),
        grid_pitch_mm=0.2,
        config=RoughnessConfig(grid_mm=0.2),
    )
    rep_pass = format_report(res, target_svr_mm=0.05)
    assert "PASS (Within specification)" in rep_pass
    assert "VALID (>= 50x50 mm)" in rep_pass

    rep_fail = format_report(res, target_svr_mm=0.02)
    assert "FAIL (Exceeds specification)" in rep_fail


def test_save_metrics_and_grid(tmp_path: Path):
    res = RoughnessResult(
        svr_um=28.0,
        sa_um=22.0,
        sq_um=30.0,
        grid=np.ones((10, 10)),
        config=RoughnessConfig(),
    )
    json_path = tmp_path / "metrics.json"
    npz_path = tmp_path / "grid.npz"

    res.save_metrics_json(json_path)
    assert json_path.is_file()
    assert "svr_um" in json_path.read_text(encoding="utf-8")

    res.save_grid_npz(npz_path)
    assert npz_path.is_file()
    loaded_npz = np.load(npz_path)
    assert "grid" in loaded_npz
    assert loaded_npz["grid"].shape == (10, 10)

