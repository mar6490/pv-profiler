from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from pv_profiler import batch


class FakeBooleanMasks:
    def __init__(self, n: int) -> None:
        self.clear_times = pd.Series([True] * n)


class FakeDataHandler:
    def __init__(self, df: pd.DataFrame) -> None:
        self.data_frame = df.copy()
        self.boolean_masks = FakeBooleanMasks(len(df))
        self.clean_power = df["ac_power"] * 0.99

    def run_pipeline(self, power_col: str, fix_shifts: bool, verbose: bool) -> None:
        _ = (power_col, fix_shifts, verbose)

    def augment_data_frame(self, values: pd.Series, name: str) -> None:
        self.data_frame[name] = values.values

    def report(self, return_values: bool = True):
        _ = return_values
        return {"time shift correction": True, "time zone correction": 0.0}


def _fake_write_parquet(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = df.to_frame() if hasattr(df, "to_frame") and not hasattr(df, "columns") else df
    frame.to_csv(path, index=True)


def test_run_single_smoke(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setitem(__import__("sys").modules, "solardatatools", SimpleNamespace(DataHandler=FakeDataHandler))
    monkeypatch.setattr(batch, "write_parquet", _fake_write_parquet)

    input_csv = tmp_path / "example_single.csv"
    pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01T00:00:00",
                "2024-01-01T00:05:00",
                "2024-01-01T00:10:00",
            ],
            "ac_power": [100.0, 120.0, 110.0],
        }
    ).to_csv(input_csv, index=False)

    plants_csv = tmp_path / "plants.csv"
    pd.DataFrame(
        {
            "system_id": ["sys1"],
            "country": ["DE"],
            "plz": ["12345"],
            "lat": [52.5],
            "lon": [13.4],
            "timezone": ["Europe/Berlin"],
        }
    ).to_csv(plants_csv, index=False)

    cfg = {
        "paths": {"output_root": str(tmp_path / "outputs"), "plants_csv": str(plants_csv)},
        "pipeline": {"fit_tau": 0.03},
    }

    result = batch.run_single(system_id="sys1", input_path=str(input_csv), config=cfg, lat=None, lon=None)
    assert result["system_id"] == "sys1"

    run_dir = next((tmp_path / "outputs" / "sys1").glob("*"))
    expected = [
        "01_parsed_tzaware.parquet",
        "02_cleaned_timeshift_fixed.parquet",
        "03_clear_times_mask.parquet",
        "04_daily_flags.csv",
        "05_clipping_summary.json",
        "06_clipped_times_mask.parquet",
        "07_sdt_summary.json",
        "07_sdt_introspect.json",
        "08_daily_peak.csv",
        "09_p_norm.parquet",
        "11_fit_mask.parquet",
        "12_daily_fit_fraction.csv",
        "summary.json",
    ]
    for filename in expected:
        assert (run_dir / filename).exists(), filename
