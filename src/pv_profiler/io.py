"""Input/output helpers for PV profiler."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pv_profiler.validation import parse_and_validate_timestamp_index, validate_power_series, validate_regular_sampling


def _read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported file format: {suffix}")


def read_single_plant(path: str | Path, timestamp_col: str = "timestamp", power_col: str = "ac_power") -> pd.DataFrame:
    """Load single-plant time series and validate contract."""
    df = _read_table(path)
    if timestamp_col not in df.columns:
        raise ValueError(f"Missing timestamp column '{timestamp_col}'.")
    validate_power_series(df, power_col)
    index = parse_and_validate_timestamp_index(df[timestamp_col])
    out = pd.DataFrame({power_col: pd.to_numeric(df[power_col], errors="coerce")}, index=index)
    out = out.sort_index()
    validate_regular_sampling(out.index)
    return out


def read_wide_plants(path: str | Path, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Load wide multi-system file and validate the shared timestamp index."""
    df = _read_table(path)
    if timestamp_col not in df.columns:
        raise ValueError(f"Missing timestamp column '{timestamp_col}'.")
    index = parse_and_validate_timestamp_index(df[timestamp_col])
    wide = df.drop(columns=[timestamp_col]).apply(pd.to_numeric, errors="coerce")
    wide.index = index
    wide = wide.sort_index()
    validate_regular_sampling(wide.index)
    return wide


def read_manifest(path: str | Path) -> pd.DataFrame:
    """Load manifest CSV with required 'system_id' column."""
    manifest = pd.read_csv(path)
    if "system_id" not in manifest.columns:
        raise ValueError("manifest.csv must contain a 'system_id' column.")
    return manifest


def read_plants_metadata(path: str | Path) -> pd.DataFrame:
    """Load plant metadata table used for system_id -> location lookup."""
    meta = pd.read_csv(path)
    required = {"system_id", "lat", "lon"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"plants metadata missing columns: {sorted(missing)}")
    return meta


def write_parquet(df: pd.DataFrame | pd.Series, path: str | Path) -> None:
    """Write parquet using pandas with optional engine detection."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    frame = df.to_frame() if isinstance(df, pd.Series) else df
    frame.to_parquet(p)


def write_json(payload: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
