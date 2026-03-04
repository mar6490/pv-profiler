from __future__ import annotations

import json

import pandas as pd

from pv_profiler.benchmark import build_benchmark_results, parse_two_azimuths


def test_parse_two_azimuths_slash_format():
    a, b = parse_two_azimuths("270.000000/90.000000")
    assert a == 270.0
    assert b == 90.0


def test_build_benchmark_results_handles_east_west_azimuth_string(tmp_path):
    out_root = tmp_path / "out"
    sys_dir = out_root / "system_001"
    sys_dir.mkdir(parents=True)

    result = {
        "model_type": "two_plane",
        "tilt_deg": 25.0,
        "azimuth_center_deg": 180.0,
        "azimuth_east_deg": 90.0,
        "azimuth_west_deg": 270.0,
        "weight_east": 0.4,
        "score_rmse": 0.1,
        "score_bic": -2.0,
        "timing_seconds": {"total": 1.2},
    }
    (sys_dir / "08_orientation_result.json").write_text(json.dumps(result), encoding="utf-8")

    meta = pd.DataFrame(
        [
            {
                "system_id": 1,
                "system_type": "east-west",
                "tilt": 25.0,
                "azimuth": "270.000000/90.000000",
                "azimuth_center_deg_true": 180.0,
                "weight_true": 0.5,
            }
        ]
    )
    meta_path = tmp_path / "systems_metadata.csv"
    meta.to_csv(meta_path, index=False)

    df = build_benchmark_results(out_root, meta_path)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["status"] == "ok"
    assert pd.notna(row["az_center_abs_err_deg"])
    assert pd.notna(row["az_plane_abs_err_deg"])
    assert pd.notna(row["weight_abs_err"])


def test_plane_matching_uses_min_assignment(tmp_path):
    out_root = tmp_path / "out"
    sys_dir = out_root / "system_002"
    sys_dir.mkdir(parents=True)

    # swapped east/west estimates should still achieve 0 plane error via min matching
    result = {
        "model_type": "two_plane",
        "tilt_deg": 20.0,
        "azimuth_center_deg": 180.0,
        "azimuth_east_deg": 270.0,
        "azimuth_west_deg": 90.0,
        "weight_east": 0.5,
    }
    (sys_dir / "08_orientation_result.json").write_text(json.dumps(result), encoding="utf-8")

    meta = pd.DataFrame(
        [
            {
                "system_id": 2,
                "system_type": "east_west",
                "tilt": 20.0,
                "azimuth_east_deg_true": 90.0,
                "azimuth_west_deg_true": 270.0,
            }
        ]
    )
    meta_path = tmp_path / "systems_metadata.csv"
    meta.to_csv(meta_path, index=False)

    df = build_benchmark_results(out_root, meta_path)
    row = df.iloc[0]
    assert row["az_plane_abs_err_deg"] == 0.0
