from __future__ import annotations

from pathlib import Path

import pandas as pd

from pv_profiler.io import read_single_plant, read_wide_plants


def test_read_single_plant_csv(tmp_path: Path) -> None:
    path = tmp_path / "single.csv"
    pd.DataFrame(
        {
            "timestamp": ["2024-01-01T00:00:00", "2024-01-01T00:05:00"],
            "ac_power": [1.0, 2.0],
        }
    ).to_csv(path, index=False)

    df = read_single_plant(path)
    assert "ac_power" in df.columns
    assert str(df.index.tz) == "Etc/GMT-1"


def test_read_wide_plants_csv(tmp_path: Path) -> None:
    path = tmp_path / "wide.csv"
    pd.DataFrame(
        {
            "timestamp": ["2024-01-01T00:00:00", "2024-01-01T00:05:00"],
            "s1": [1.0, 2.0],
            "s2": [3.0, 4.0],
        }
    ).to_csv(path, index=False)

    df = read_wide_plants(path)
    assert list(df.columns) == ["s1", "s2"]


def test_read_single_plant_parquet_with_datetime_index(tmp_path: Path) -> None:
    path = tmp_path / "single_index.parquet"
    idx = pd.date_range("2024-01-01T00:00:00", periods=2, freq="5min")
    pd.DataFrame({"ac_power": [1.0, 2.0]}, index=idx).to_parquet(path)

    df = read_single_plant(path)
    assert "ac_power" in df.columns
    assert str(df.index.tz) == "Etc/GMT-1"
