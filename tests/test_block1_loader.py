from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pv_profiler.block_io import load_input_for_sdt
from pv_profiler.cli import main


DATA_CSV = Path("data/sonnja_pv3_2015/einleuchtend_wrdata_2015_wr1_5min_naive.csv")


def test_block1_loader_standardizes_output_and_sampling():
    result = load_input_for_sdt(
        input_path=DATA_CSV,
        timestamp_col="timestamp",
        power_col="P_AC",
        min_samples=100,
    )

    assert list(result.data.columns) == ["power"]
    assert result.data.index.is_monotonic_increasing
    assert result.diagnostics.dominant_timedelta is not None
    assert result.diagnostics.share_non_null_power > 0.5


def test_block1_loader_reads_only_required_columns(monkeypatch):
    calls: dict[str, object] = {}
    real_read_csv = pd.read_csv

    def fake_read_csv(*args, **kwargs):
        calls.update(kwargs)
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr("pv_profiler.block_io.pd.read_csv", fake_read_csv)

    load_input_for_sdt(DATA_CSV, timestamp_col="timestamp", power_col="P_AC", min_samples=100)

    assert calls.get("usecols") == ["timestamp", "P_AC"]


def test_cli_run_block1_writes_required_artifacts(monkeypatch, tmp_path):
    out_dir = tmp_path / "block1"
    monkeypatch.setattr(
        "sys.argv",
        [
            "pv-ident",
            "run-block1",
            "--input-csv",
            str(DATA_CSV),
            "--output-dir",
            str(out_dir),
            "--timestamp-col",
            "timestamp",
            "--power-col",
            "P_AC",
            "--min-samples",
            "100",
        ],
    )

    rc = main()

    assert rc == 0
    parquet_path = out_dir / "01_input_power.parquet"
    diagnostics_path = out_dir / "01_input_diagnostics.json"
    assert parquet_path.exists()
    assert diagnostics_path.exists()

    diag = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert "shape" in diag
    assert "sampling_summary" in diag
    assert "decisions" in diag


def test_block1_loader_tz_naive_localizes_without_error(tmp_path):
    csv_path = tmp_path / "naive.csv"
    csv_path.write_text(
        "time,ac_power_w\n"
        "2025-01-01 00:00:00,10\n"
        "2025-01-01 00:05:00,11\n"
        "2025-01-01 00:10:00,12\n",
        encoding="utf-8",
    )

    result = load_input_for_sdt(
        input_path=csv_path,
        timestamp_col="time",
        power_col="ac_power_w",
        timezone="Etc/GMT-1",
        min_samples=3,
    )

    assert result.data.index[0].strftime("%Y-%m-%d %H:%M:%S") == "2025-01-01 00:00:00"


def test_block1_loader_tz_aware_does_not_double_localize_or_shift(tmp_path):
    csv_path = tmp_path / "aware.csv"
    csv_path.write_text(
        "time,ac_power_w\n"
        "2025-01-01 00:00:00+01:00,10\n"
        "2025-01-01 00:05:00+01:00,11\n"
        "2025-01-01 00:10:00+01:00,12\n",
        encoding="utf-8",
    )

    result = load_input_for_sdt(
        input_path=csv_path,
        timestamp_col="time",
        power_col="ac_power_w",
        timezone="Etc/GMT-1",
        min_samples=3,
    )

    assert result.data.index[0].strftime("%Y-%m-%d %H:%M:%S") == "2025-01-01 00:00:00"
