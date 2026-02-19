from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pv_profiler.block_fit import load_daily_flags
from pv_profiler.pipeline import run_block3_from_files


def _power_df(days: int = 12) -> pd.DataFrame:
    idx = pd.date_range("2015-01-01", periods=days * 288, freq="5min")
    return pd.DataFrame({"power": [1.0] * len(idx)}, index=idx)


def _flags_df(days: int, fit_days: int) -> pd.DataFrame:
    idx = pd.date_range("2015-01-01", periods=days, freq="D")
    clear = [True] * fit_days + [False] * (days - fit_days)
    return pd.DataFrame(
        {
            "capacity_cluster": [2] * days,
            "clear": clear,
            "cloudy": [not x for x in clear],
            "density": [True] * days,
            "inverter_clipped": [False] * days,
            "linearity": [True] * days,
            "no_errors": [True] * days,
        },
        index=idx,
    )


def test_load_daily_flags_parses_unnamed_index_csv(tmp_path):
    flags = _flags_df(days=3, fit_days=2)
    csv_path = tmp_path / "02_sdt_daily_flags.csv"
    flags.to_csv(csv_path, index=True)  # unnamed index on purpose

    loaded = load_daily_flags(csv_path)

    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert "is_fit_day" in loaded.columns
    assert int(loaded["is_fit_day"].sum()) == 2


def test_block3_gate_fail_and_pass(tmp_path):
    power_parquet = tmp_path / "01_input_power.parquet"
    _power_df(days=12).to_parquet(power_parquet)

    flags9 = _flags_df(days=12, fit_days=9)
    flags9_csv = tmp_path / "flags9.csv"
    flags9.to_csv(flags9_csv, index=True)

    out_fail = tmp_path / "out_fail"
    res_fail = run_block3_from_files(power_parquet, flags9_csv, out_fail, min_fit_days=10)
    assert res_fail.status == "insufficient_fit_days"
    assert not (out_fail / "05_power_fit.parquet").exists()

    flags10 = _flags_df(days=12, fit_days=10)
    flags10_csv = tmp_path / "flags10.csv"
    flags10.to_csv(flags10_csv, index=True)

    out_pass = tmp_path / "out_pass"
    res_pass = run_block3_from_files(power_parquet, flags10_csv, out_pass, min_fit_days=10)
    assert res_pass.status == "ok"
    assert (out_pass / "05_power_fit.parquet").exists()


def test_block3_join_mask_to_nan_per_day(tmp_path):
    power = _power_df(days=2)
    power_parquet = tmp_path / "01_input_power.parquet"
    power.to_parquet(power_parquet)

    flags = pd.DataFrame(
        {
            "capacity_cluster": [2, 2],
            "clear": [True, False],
            "cloudy": [False, True],
            "density": [True, True],
            "inverter_clipped": [False, False],
            "linearity": [True, True],
            "no_errors": [True, True],
        },
        index=pd.to_datetime(["2015-01-01", "2015-01-02"]),
    )
    flags_csv = tmp_path / "02_sdt_daily_flags.csv"
    flags.to_csv(flags_csv, index=True)

    out_dir = tmp_path / "out"
    run_block3_from_files(power_parquet, flags_csv, out_dir, fit_mode="mask_to_nan", min_fit_days=1)

    fit = pd.read_parquet(out_dir / "05_power_fit.parquet")
    day1 = fit.loc["2015-01-01", "power"]
    day2 = fit.loc["2015-01-02", "power"]
    assert day1.notna().all()
    assert day2.isna().all()
