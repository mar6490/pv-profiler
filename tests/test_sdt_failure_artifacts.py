from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from pv_profiler.sdt_pipeline import run_block_a


class FakeMasks:
    def __init__(self, n: int) -> None:
        self.daytime = pd.Series([True] * n).to_numpy().reshape(n, 1)
        self.missing_values = pd.Series([False] * n).to_numpy().reshape(n, 1)
        self.infill = pd.Series([False] * n).to_numpy().reshape(n, 1)


class FakeDataHandler:
    def __init__(self, df: pd.DataFrame) -> None:
        self.data_frame = df.copy()
        self.boolean_masks = FakeMasks(len(df))
        self.filled_data_matrix = self.data_frame[["ac_power"]].to_numpy()
        self.daily_flags = SimpleNamespace(clear_day=[True])

    def run_pipeline(self, power_col: str, fix_shifts: bool, verbose: bool) -> None:
        _ = (power_col, fix_shifts, verbose)

    def augment_data_frame(self, values, name: str) -> None:
        import numpy as np
        self.data_frame[name] = np.asarray(values).reshape(-1)

    def report(self, return_values: bool = True):
        _ = return_values
        return {"time shift correction": False, "time zone correction": 0.0}


def test_block_a_writes_debug_artifacts_on_extract_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(__import__("sys").modules, "solardatatools", SimpleNamespace(DataHandler=FakeDataHandler))

    import pv_profiler.sdt_pipeline as mod

    def _raise(*args, **kwargs):
        _ = (args, kwargs)
        raise RuntimeError("forced extraction failure")

    monkeypatch.setattr(mod, "extract_clean_power_series", _raise)

    idx = pd.date_range("2024-01-01", periods=6, freq="5min", tz="Etc/GMT-1")
    ac = pd.Series([0, 0, 1, 2, 1, 0], index=idx)

    with pytest.raises(RuntimeError, match="forced extraction failure"):
        run_block_a(ac_power=ac, lat=52.5, lon=13.4, out_dir=tmp_path)

    assert (tmp_path / "01_parsed_tzaware.parquet").exists()
    assert (tmp_path / "07_sdt_introspect.json").exists()
