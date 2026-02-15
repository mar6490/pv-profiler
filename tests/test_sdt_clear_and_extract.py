from __future__ import annotations

import numpy as np
import pandas as pd

from pv_profiler.sdt_pipeline import apply_exclusion_rules, extract_clean_power_series, get_clear_day_flags


class FakeDailyFlags:
    def __init__(self, clear_days: np.ndarray) -> None:
        self.clear_days = clear_days


class FakeHandlerDailyFlags:
    def __init__(self, n_days: int) -> None:
        self.filled_data_matrix = np.zeros((3, n_days), dtype=float)
        self.daily_flags = FakeDailyFlags(np.array([True, False], dtype=bool))


class FakeHandlerGetDailyFlags:
    def __init__(self, n_days: int) -> None:
        self.filled_data_matrix = np.zeros((3, n_days), dtype=float)

    def get_daily_flags(self):
        return {"clear_day": np.array([False, True], dtype=bool)}


class FakeHandlerExtract:
    def __init__(self, idx: pd.DatetimeIndex) -> None:
        self.data_frame = pd.DataFrame({"ac_power": [1.0, 2.0, 3.0], "seq_index": [0, 1, 2]}, index=idx)
        self.day_index = pd.DatetimeIndex([idx[0].normalize()])
        self.filled_data_matrix = np.array([[1.0], [2.0], [3.0]])

    def report(self, return_values: bool = True):
        _ = return_values
        return {"time shift correction": True, "time zone correction": 1}


class FakeHandlerExtractCleanCol:
    def __init__(self, idx: pd.DatetimeIndex) -> None:
        self.data_frame = pd.DataFrame({"ac_power_clean": [1.0, 2.0, 3.0], "seq_index": [0, 1, 2]}, index=idx)

    def report(self, return_values: bool = True):
        _ = return_values
        return {"time shift correction": True, "time zone correction": 1}


class FakeHandlerNoCorrections:
    def __init__(self, idx: pd.DatetimeIndex) -> None:
        self.data_frame = pd.DataFrame({"ac_power": [5.0, 6.0, 7.0], "seq_index": [0, 1, 2]}, index=idx)

    def report(self, return_values: bool = True):
        _ = return_values
        return {"time shift correction": False, "time zone correction": 0}


def test_get_clear_day_flags_from_get_daily_flags() -> None:
    handler = FakeHandlerGetDailyFlags(n_days=2)
    flags, source = get_clear_day_flags(handler)
    assert flags is not None
    assert flags.tolist() == [False, True]
    assert source == "sdt:get_daily_flags:clear_day"


def test_get_clear_day_flags_from_daily_flags_attr() -> None:
    handler = FakeHandlerDailyFlags(n_days=2)
    flags, source = get_clear_day_flags(handler)
    assert flags is not None
    assert flags.tolist() == [True, False]
    assert source == "sdt:daily_flags:clear_days"


def test_apply_exclusion_rules_uses_n_fit_times() -> None:
    summary = {
        "clipping_day_share_mean": 0.0,
        "clipping_fraction_day_median": 0.0,
        "n_fit_times": 0,
        "time_shift_correction_applied": False,
        "time_zone_correction": 0.0,
    }
    result = apply_exclusion_rules(summary, config={"pipeline": {"clear_time_fraction_min": 0.005}})
    assert result["exclude_low_clear"] is True


def test_extract_clean_power_series_works_with_introspected_attributes() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="5min", tz="Etc/GMT-1")
    handler = FakeHandlerExtract(idx)

    series, source = extract_clean_power_series(handler, power_col="ac_power")
    assert isinstance(series, pd.Series)
    assert series.name == "ac_power_clean"
    assert list(series.values) == [1.0, 2.0, 3.0]
    assert "filled_data_matrix" in source


def test_extract_prefers_clean_post_pipeline_column() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="5min", tz="Etc/GMT-1")
    handler = FakeHandlerExtractCleanCol(idx)

    series, source = extract_clean_power_series(handler, power_col="ac_power")
    assert list(series.values) == [1.0, 2.0, 3.0]
    assert "ac_power_clean" in source


def test_extract_returns_power_col_when_no_corrections() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="5min", tz="Etc/GMT-1")
    handler = FakeHandlerNoCorrections(idx)

    series, source = extract_clean_power_series(handler, power_col="ac_power")
    assert list(series.values) == [5.0, 6.0, 7.0]
    assert source == "data_frame:ac_power:no_corrections"
