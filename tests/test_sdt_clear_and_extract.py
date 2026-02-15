from __future__ import annotations

import numpy as np
import pandas as pd

from pv_profiler.sdt_pipeline import _ensure_clear_times, extract_clean_power_series


class FakeBooleanMasks:
    def __init__(self, values: pd.Series) -> None:
        self.clear_times = values


class FakeHandlerClear:
    def __init__(self, idx: pd.DatetimeIndex) -> None:
        self.data_frame = pd.DataFrame({"ac_power": [0.0] * len(idx), "seq_index": np.arange(len(idx))}, index=idx)
        self.boolean_masks = FakeBooleanMasks(pd.Series([True] * len(idx), index=idx))


class FakeHandlerExtract:
    def __init__(self, idx: pd.DatetimeIndex) -> None:
        self.data_frame = pd.DataFrame({"ac_power": [1.0, 2.0, 3.0], "seq_index": [0, 1, 2]}, index=idx)
        self.day_index = pd.DatetimeIndex([idx[0].normalize()])
        self.filled_data_matrix = np.array([[1.0], [2.0], [3.0]])


def test_clear_times_available_after_block_a_step() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="5min", tz="Etc/GMT-1")
    handler = FakeHandlerClear(idx)
    mask, source = _ensure_clear_times(handler)

    assert isinstance(mask, pd.Series)
    assert mask.name is None or mask.name == "is_clear_time" or mask.name == "clear_times"
    assert mask.astype(bool).all()
    assert source.startswith("dh.boolean_masks")


def test_extract_clean_power_series_works_with_introspected_attributes() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="5min", tz="Etc/GMT-1")
    handler = FakeHandlerExtract(idx)

    series, source = extract_clean_power_series(handler, power_col="ac_power")
    assert isinstance(series, pd.Series)
    assert series.name == "ac_power_clean"
    assert list(series.values) == [1.0, 2.0, 3.0]
    assert "dh.data_frame" in source
