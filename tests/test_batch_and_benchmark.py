from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pv_profiler.batch import parse_system_id_from_filename, run_batch
from pv_profiler.benchmark import circular_err_deg


def test_circular_err_deg():
    assert circular_err_deg(10, 350) == 20
    assert circular_err_deg(350, 10) == 20
    assert circular_err_deg(90, 90) == 0


def test_parse_system_id_from_filename():
    assert parse_system_id_from_filename(Path("system_001.csv")) == 1


def test_run_batch_metadata_join_uses_int_system_id(monkeypatch, tmp_path):
    in_dir = tmp_path / "inputs"
    out_root = tmp_path / "out"
    in_dir.mkdir()

    for sid in ["system_001", "system_002"]:
        (in_dir / f"{sid}.csv").write_text("time,ac_power_w\n2015-01-01 00:00:00,0\n", encoding="utf-8")

    meta = tmp_path / "systems_metadata.csv"
    pd.DataFrame({"system_id": [1], "lat": [52.45], "lon": [13.52]}).to_csv(meta, index=False)

    def fake_run_blocks_for_system(**kwargs):
        out_dir = kwargs["output_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "model_type": "single",
            "tilt_deg": 20.0,
            "azimuth_deg": 180.0,
            "score_rmse": 0.1,
            "score_bic": -10.0,
        }
        (out_dir / "08_orientation_result.json").write_text(json.dumps(result), encoding="utf-8")
        return result

    monkeypatch.setattr("pv_profiler.batch._run_blocks_for_system", fake_run_blocks_for_system)

    df = run_batch(
        input_dir=in_dir,
        pattern="system_*.csv",
        output_root=out_root,
        timestamp_col="time",
        power_col="ac_power_w",
        timezone_str="Etc/GMT-1",
        systems_metadata_csv=meta,
        system_id_col="system_id",
        jobs=1,
        skip_existing=False,
    )

    status_map = {int(r["system_id"]): r["status"] for _, r in df.iterrows()}
    assert status_map[1] == "ok"
    assert status_map[2] == "failed"

    failed_status = json.loads((out_root / "system_002" / "00_status.json").read_text(encoding="utf-8"))
    assert failed_status["status"] == "failed"
    assert "parsed system_id=2" in failed_status["error"]["message"]


def test_run_batch_smoke_and_skip_existing(monkeypatch, tmp_path):
    in_dir = tmp_path / "inputs"
    out_root = tmp_path / "out"
    in_dir.mkdir()

    for sid in ["system_001", "system_002"]:
        (in_dir / f"{sid}.csv").write_text("time,ac_power_w\n2015-01-01 00:00:00,0\n", encoding="utf-8")

    def fake_run_blocks_for_system(**kwargs):
        out_dir = kwargs["output_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "model_type": "single",
            "tilt_deg": 20.0,
            "azimuth_deg": 180.0,
            "score_rmse": 0.1,
            "score_bic": -10.0,
        }
        (out_dir / "08_orientation_result.json").write_text(json.dumps(result), encoding="utf-8")
        return result

    monkeypatch.setattr("pv_profiler.batch._run_blocks_for_system", fake_run_blocks_for_system)

    df1 = run_batch(
        input_dir=in_dir,
        pattern="system_*.csv",
        output_root=out_root,
        timestamp_col="time",
        power_col="ac_power_w",
        timezone_str="Etc/GMT-1",
        latitude=52.45,
        longitude=13.52,
        jobs=1,
        skip_existing=False,
    )

    assert set(df1["status"]) == {"ok"}
    assert (out_root / "batch_summary.csv").exists()
    assert (out_root / "system_001" / "00_status.json").exists()

    df2 = run_batch(
        input_dir=in_dir,
        pattern="system_*.csv",
        output_root=out_root,
        timestamp_col="time",
        power_col="ac_power_w",
        timezone_str="Etc/GMT-1",
        latitude=52.45,
        longitude=13.52,
        jobs=1,
        skip_existing=True,
    )

    assert set(df2["status"]) == {"skipped"}
