"""Input/output helpers for PV profiler."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pv_profiler.validation import parse_and_validate_timestamp_index, validate_power_series, validate_regular_sampling


def _resolve_timestamp_values(df: pd.DataFrame, timestamp_col: str) -> pd.Series | pd.Index:
    """Resolve timestamp source from explicit column, case-insensitive column, or DatetimeIndex."""
    if timestamp_col in df.columns:
        return df[timestamp_col]

    lowered = {str(c).lower(): c for c in df.columns}
    key = timestamp_col.lower()
    if key in lowered:
        return df[lowered[key]]

    if isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(df.index, index=df.index)

    raise ValueError(f"Missing timestamp column '{timestamp_col}'.")


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
    timestamp_values = _resolve_timestamp_values(df, timestamp_col)
    validate_power_series(df, power_col)
    index = parse_and_validate_timestamp_index(timestamp_values)
    out = pd.DataFrame({"ac_power": pd.to_numeric(df[power_col], errors="coerce")}, index=index)
    out = out.sort_index()
    validate_regular_sampling(out.index)
    return out


def read_single_plant_notebook_csv(path: str | Path) -> pd.DataFrame:
    """Load known-good SDT CSV exactly like the notebook reference."""
    df = pd.read_csv(
        path,
        sep=",",
        quotechar='"',
        encoding="utf-8-sig",
        usecols=["timestamp", "P_AC"],
        low_memory=False,
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

    power = df[["P_AC"]].rename(columns={"P_AC": "ac_power"})
    power["ac_power"] = pd.to_numeric(power["ac_power"], errors="coerce").fillna(0).clip(lower=0)
    return power


def read_wide_plants(path: str | Path, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Load wide multi-system file and validate the shared timestamp index."""
    df = _read_table(path)
    timestamp_values = _resolve_timestamp_values(df, timestamp_col)
    index = parse_and_validate_timestamp_index(timestamp_values)

    if timestamp_col in df.columns:
        wide_source = df.drop(columns=[timestamp_col])
    else:
        lowered = {str(c).lower(): c for c in df.columns}
        if timestamp_col.lower() in lowered:
            wide_source = df.drop(columns=[lowered[timestamp_col.lower()]])
        else:
            wide_source = df

    wide = wide_source.apply(pd.to_numeric, errors="coerce")
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


def _find_first_number(payload: Any, keys: tuple[str, ...]) -> float | None:
    if isinstance(payload, dict):
        for key in keys:
            if key in payload:
                try:
                    return float(payload[key])
                except (TypeError, ValueError):
                    pass
        for value in payload.values():
            found = _find_first_number(value, keys)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_first_number(item, keys)
            if found is not None:
                return found
    return None


def read_metadata_json(path: str | Path) -> dict[str, float | None]:
    """Load lat/lon/altitude from metadata JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    lat = _find_first_number(payload, ("lat", "latitude"))
    lon = _find_first_number(payload, ("lon", "lng", "longitude"))
    alt = _find_first_number(payload, ("altitude", "elevation", "alt"))
    if lat is None or lon is None:
        raise ValueError("metadata JSON must contain latitude and longitude fields.")
    return {"lat": lat, "lon": lon, "altitude": alt}


def write_parquet(df: pd.DataFrame | pd.Series, path: str | Path) -> None:
    """Write parquet using pandas with optional engine detection."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    frame = df.to_frame() if isinstance(df, pd.Series) else df
    frame.to_parquet(p)


def to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (datetime, date, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, pd.Series):
        return [to_jsonable(v) for v in obj.to_list()]
    if isinstance(obj, pd.DataFrame):
        return {str(k): to_jsonable(v) for k, v in obj.to_dict(orient="list").items()}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    return str(obj)


def write_json(payload: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload2 = to_jsonable(payload)
    p.write_text(json.dumps(payload2, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
