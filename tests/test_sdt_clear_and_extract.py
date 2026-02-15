from __future__ import annotations

import numpy as np
import pandas as pd

from pv_profiler.sdt_pipeline import _ensure_clear_times, extract_clean_power_series


class FakeBooleanMasks:
    def __init__(self, values) -> None:
        self.clear_times = values


class FakeHandlerClear:
    def __init__(self, idx: pd.DatetimeIndex) -> None:
        self.data_frame = pd.DataFrame({"ac_power": [0.0] * len(idx), "seq_index": np.arange(len(idx))}, index=idx)
        self.boolean_masks = FakeBooleanMasks(pd.Series([True] * len(idx), index=idx))

    def augment_data_frame(self, values, name: str):
        self.data_frame[name] = pd.Series(values, index=self.data_frame.index).values


class FakeHandlerClearNumpy:
    def __init__(self, idx: pd.DatetimeIndex) -> None:
        self.data_frame = pd.DataFrame({"ac_power": [0.0] * len(idx), "seq_index": np.arange(len(idx))}, index=idx)
        self.boolean_masks = FakeBooleanMasks(np.array([True] * len(idx), dtype=bool))

    def augment_data_frame(self, values, name: str):
        self.data_frame[name] = np.asarray(values)


class FakeHandlerClearViaMethod:
    def __init__(self, idx: pd.DatetimeIndex) -> None:
        self.data_frame = pd.DataFrame({"ac_power": [0.0] * len(idx), "seq_index": np.arange(len(idx))}, index=idx)
        self.boolean_masks = FakeBooleanMasks(None)

    def make_filled_data_matrix(self):
        return None

    def find_clipped_times(self):
        return None

    def get_daily_flags(self):
        return None

    def calculate_scsf_performance_index(self):
        return None

    def find_clear_times(self):
        self.boolean_masks.clear_times = pd.Series([True] * len(self.data_frame), index=self.data_frame.index)

    def augment_data_frame(self, values, name: str):
        self.data_frame[name] = pd.Series(values, index=self.data_frame.index).values


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


def test_clear_times_available_after_block_a_step() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="5min", tz="Etc/GMT-1")
    handler = FakeHandlerClear(idx)
    mask, source = _ensure_clear_times(handler)

    assert isinstance(mask, pd.Series)
    assert mask.astype(bool).all()
    assert len(mask) == len(idx)
    assert source.startswith("dh.boolean_masks")


def test_clear_times_numpy_array_same_length() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="5min", tz="Etc/GMT-1")
    handler = FakeHandlerClearNumpy(idx)

    mask, source = _ensure_clear_times(handler)
    assert isinstance(mask, pd.Series)
    assert len(mask) == len(handler.data_frame.index)
    assert mask.astype(bool).all()
    assert source.startswith("dh.boolean_masks")


def test_clear_times_computed_via_public_method() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="5min", tz="Etc/GMT-1")
    handler = FakeHandlerClearViaMethod(idx)

    mask, source = _ensure_clear_times(handler)
    assert mask.astype(bool).all()
    assert source.startswith("dh.boolean_masks")


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
