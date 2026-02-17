from __future__ import annotations

import pandas as pd

from pv_profiler.block_normalization import run_block4_normalize_from_parquet


def _write_power_fit(path, values, idx=None):
    if idx is None:
        idx = pd.date_range("2015-01-01", periods=len(values), freq="5min")
    df = pd.DataFrame({"power": values}, index=idx)
    df.to_parquet(path)


def test_daily_peak_uses_fit_samples_and_handles_empty_day(tmp_path):
    day1 = [0.0, 1.0, 2.0, 3.0]
    day2 = [float("nan")] * 4
    values = day1 + day2
    idx = list(pd.date_range("2015-01-01 00:00:00", periods=4, freq="5min")) + list(
        pd.date_range("2015-01-02 00:00:00", periods=4, freq="5min")
    )

    p = tmp_path / "05_power_fit.parquet"
    _write_power_fit(p, values, idx=idx)

    res = run_block4_normalize_from_parquet(
        input_power_fit_parquet=p,
        output_dir=tmp_path / "out",
        quantile=0.995,
        min_fit_samples_day=1,
        dropna_output=False,
    )

    day2_key = pd.Timestamp("2015-01-01") + pd.Timedelta(days=1)
    assert bool(res.daily_peak.loc[day2_key, "is_usable_day"]) is False
    assert pd.isna(res.p_norm.loc[res.p_norm.index.normalize() == day2_key, "p_norm"]).all()


def test_peak_le_zero_not_usable(tmp_path):
    values = [0.0] * 10
    p = tmp_path / "05_power_fit.parquet"
    _write_power_fit(p, values)

    res = run_block4_normalize_from_parquet(
        input_power_fit_parquet=p,
        output_dir=tmp_path / "out",
        quantile=0.995,
        min_fit_samples_day=1,
        dropna_output=False,
    )

    first_day = res.daily_peak.index[0]
    assert res.daily_peak.loc[first_day, "p_peak_day"] == 0.0
    assert bool(res.daily_peak.loc[first_day, "is_usable_day"]) is False


def test_dropna_output_only_non_null(tmp_path):
    values = [1.0, 2.0, float("nan"), 3.0, float("nan")]
    p = tmp_path / "05_power_fit.parquet"
    _write_power_fit(p, values)

    out_dir = tmp_path / "out"
    run_block4_normalize_from_parquet(
        input_power_fit_parquet=p,
        output_dir=out_dir,
        quantile=0.995,
        min_fit_samples_day=1,
        dropna_output=True,
    )

    out = pd.read_parquet(out_dir / "07_p_norm_clear.parquet")
    assert out["p_norm"].notna().all()


def test_daily_peak_csv_has_date_column_with_iso_format_for_named_index(tmp_path):
    idx = pd.date_range("2015-01-01", periods=6, freq="5min")
    df = pd.DataFrame({"power": [1.0, 2.0, 1.5, 2.5, 1.8, 2.2]}, index=idx)
    df.index.name = "timestamp"

    p = tmp_path / "05_power_fit.parquet"
    df.to_parquet(p)

    out_dir = tmp_path / "out"
    run_block4_normalize_from_parquet(
        input_power_fit_parquet=p,
        output_dir=out_dir,
        quantile=0.995,
        min_fit_samples_day=1,
        dropna_output=True,
    )

    daily_peak_csv = pd.read_csv(out_dir / "06_daily_peak.csv")
    assert "date" in daily_peak_csv.columns
    assert daily_peak_csv["date"].astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()
