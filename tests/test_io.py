from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pv_profiler.io import read_single_plant, read_wide_plants, write_json


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


def test_write_json_handles_numpy_pandas_path_types(tmp_path: Path) -> None:
    out = tmp_path / "payload.json"
    payload = {
        "flag": np.bool_(True),
        "count": np.int64(1),
        "arr": np.array([1, 2]),
        "ts": pd.Timestamp("2024-01-01T00:00:00"),
        "path": tmp_path / "x.txt",
    }

    write_json(payload, out)

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["flag"] is True
    assert loaded["count"] == 1
    assert loaded["arr"] == [1, 2]
    assert loaded["ts"].startswith("2024-01-01T00:00:00")
    assert loaded["path"].endswith("x.txt")
