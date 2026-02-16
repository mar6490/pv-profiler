from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .types import TimeSeriesData


def read_power_timeseries(csv_path: str | Path, power_column: str = "P_AC") -> TimeSeriesData:
    df = pd.read_csv(csv_path, low_memory=False)
    if "timestamp" not in df.columns:
        raise ValueError("Expected column 'timestamp' in CSV.")
    if power_column not in df.columns:
        raise ValueError(f"Expected column '{power_column}' in CSV.")

    series = pd.to_numeric(df[power_column], errors="coerce")
    index = pd.to_datetime(df["timestamp"], errors="coerce")
    mask = index.notna() & series.notna()

    ts = pd.Series(series[mask].to_numpy(), index=index[mask], name=power_column).sort_index()

    timezone = None
    if "tz" in df.columns:
        non_null_tz = df["tz"].dropna()
        if not non_null_tz.empty:
            timezone = str(non_null_tz.iloc[0]).replace("\\/", "/")

    if timezone and ts.index.tz is None:
        ts.index = ts.index.tz_localize(timezone, ambiguous="NaT", nonexistent="shift_forward")
        ts = ts[ts.index.notna()]

    return TimeSeriesData(data=ts, timezone=timezone, source_column=power_column)


def read_metadata(metadata_path: str | Path | None) -> dict[str, Any]:
    if metadata_path is None:
        return {}
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
