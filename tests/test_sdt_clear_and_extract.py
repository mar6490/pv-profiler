from __future__ import annotations

import numpy as np
import pandas as pd

from pv_profiler.sdt_pipeline import _ensure_clear_times, apply_exclusion_rules, extract_clean_power_series


class FakeBooleanMasks:
    def __init__(self, n_rows: int, n_days: int) -> None:
        self.clear_times = None
        self.daytime = np.ones((n_rows, n_days), dtype=bool)
        self.missing_values = np.zeros((n_rows, n_days), dtype=bool)
        self.infill = np.zeros((n_rows, n_days), dtype=bool)


class FakeHandlerClearDays:
    def __init__(self, idx: pd.DatetimeIndex, n_rows: int, n_days: int) -> None:
        self.data_frame = pd.DataFrame({"ac_power": [0.0] * len(idx), "seq_index": np.arange(len(idx))}, index=idx)
        self.filled_data_matrix = np.zeros((n_rows, n_days), dtype=float)
        self.boolean_masks = FakeBooleanMasks(n_rows=n_rows, n_days=n_days)
        self.augment_calls: list[tuple[tuple[int, ...], str]] = []

    def detect_clear_days(self):
        return np.array([True, False], dtype=bool)

    def augment_data_frame(self, values, name: str):
        arr = np.asarray(values)
        self.augment_calls.append((arr.shape, name))
        self.data_frame[name] = arr.reshape(-1)


class FakeHandlerDaytimeFallback:
    def __init__(self, idx: pd.DatetimeIndex, n_rows: int, n_days: int) -> None:
        self.data_frame = pd.DataFrame({"ac_power": [0.0] * len(idx), "seq_index": np.arange(len(idx))}, index=idx)
        self.filled_data_matrix = np.zeros((n_rows, n_days), dtype=float)
        self.boolean_masks = FakeBooleanMasks(n_rows=n_rows, n_days=n_days)
        self.augment_calls: list[tuple[tuple[int, ...], str]] = []

    def augment_data_frame(self, values, name: str):
        arr = np.asarray(values)
        self.augment_calls.append((arr.shape, name))
        self.data_frame[name] = arr.reshape(-1)


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


def test_clear_times_uses_detect_clear_days_tiled_mask_when_clear_times_none() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="5min", tz="Etc/GMT-1")
    handler = FakeHandlerClearDays(idx=idx, n_rows=3, n_days=2)

    mask, source = _ensure_clear_times(handler)

    assert isinstance(mask, pd.Series)
    assert source == "detect_clear_days:tiled&daytime&~missing"
    assert handler.augment_calls[0] == ((3, 2), "is_clear_time")
    assert mask.astype(bool).tolist() == [True, False, True, False, True, False]


def test_clear_times_falls_back_to_daytime_only_when_detect_clear_days_missing() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="5min", tz="Etc/GMT-1")
    handler = FakeHandlerDaytimeFallback(idx=idx, n_rows=2, n_days=2)

    mask, source = _ensure_clear_times(handler)

    assert source == "fallback:daytime_only"
    assert handler.augment_calls[0] == ((2, 2), "is_clear_time")
    assert mask.astype(bool).all()


def test_apply_exclusion_rules_forces_low_clear_on_daytime_only_fallback() -> None:
    summary = {
        "clipping_day_share_mean": 0.0,
        "clipping_fraction_day_median": 0.0,
        "clear_time_fraction_overall": 0.9,
        "clear_times_source": "fallback:daytime_only",
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
