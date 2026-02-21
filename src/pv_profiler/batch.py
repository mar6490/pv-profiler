from __future__ import annotations

import json
import re
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from .pipeline import (
    run_block1_input_loader,
    run_block2_sdt_from_parquet,
    run_block3_from_files,
    run_block4_from_files,
    run_block5_from_files,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _system_sort_key(path: Path) -> tuple[int, str]:
    m = re.search(r"(\d+)", path.stem)
    if m:
        return int(m.group(1)), path.name
    return 10**12, path.name


def discover_input_csvs(input_dir: str | Path, pattern: str) -> list[Path]:
    paths = sorted(Path(input_dir).glob(pattern), key=_system_sort_key)
    return [p for p in paths if p.is_file()]


def _load_lat_lon_map(systems_metadata_csv: str | Path | None, system_id_col: str) -> dict[str, tuple[float, float]]:
    if systems_metadata_csv is None:
        return {}
    df = pd.read_csv(systems_metadata_csv)
    lat_col = "lat" if "lat" in df.columns else ("latitude" if "latitude" in df.columns else None)
    lon_col = "lon" if "lon" in df.columns else ("longitude" if "longitude" in df.columns else None)
    if lat_col is None or lon_col is None or system_id_col not in df.columns:
        return {}

    out: dict[str, tuple[float, float]] = {}
    for _, row in df.iterrows():
        sid = str(row[system_id_col])
        try:
            out[sid] = (float(row[lat_col]), float(row[lon_col]))
        except Exception:
            continue
    return out


def _run_blocks_for_system(
    *,
    input_csv: Path,
    output_dir: Path,
    timestamp_col: str,
    power_col: str,
    timezone_str: str,
    latitude: float,
    longitude: float,
) -> dict[str, Any]:
    run_block1_input_loader(
        input_csv=input_csv,
        output_dir=output_dir,
        timestamp_col=timestamp_col,
        power_col=power_col,
        timezone=timezone_str,
    )
    run_block2_sdt_from_parquet(input_parquet=output_dir / "01_input_power.parquet", output_dir=output_dir)
    run_block3_from_files(
        input_power_parquet=output_dir / "01_input_power.parquet",
        input_daily_flags_csv=output_dir / "02_sdt_daily_flags.csv",
        output_dir=output_dir,
    )
    run_block4_from_files(input_power_fit_parquet=output_dir / "05_power_fit.parquet", output_dir=output_dir)
    result = run_block5_from_files(
        input_p_norm_parquet=output_dir / "07_p_norm_clear.parquet",
        output_dir=output_dir,
        latitude=latitude,
        longitude=longitude,
        timezone=timezone_str,
    )
    return result


def _process_one(
    *,
    input_csv: Path,
    output_root: Path,
    timestamp_col: str,
    power_col: str,
    timezone_str: str,
    latitude: float | None,
    longitude: float | None,
    lat_lon_map: dict[str, tuple[float, float]],
    skip_existing: bool,
) -> dict[str, Any]:
    system_id = input_csv.stem
    output_dir = output_root / system_id
    output_dir.mkdir(parents=True, exist_ok=True)

    started = _now_iso()
    t0 = perf_counter()

    result_json = output_dir / "08_orientation_result.json"
    if skip_existing and result_json.exists():
        payload = {
            "system_id": system_id,
            "input_csv": str(input_csv),
            "status": "skipped",
            "error": None,
            "started_at": started,
            "finished_at": _now_iso(),
            "runtime_seconds": perf_counter() - t0,
        }
        (output_dir / "00_status.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return {**payload, "model_type": None}

    try:
        if latitude is not None and longitude is not None:
            lat, lon = latitude, longitude
        else:
            if system_id not in lat_lon_map:
                raise ValueError(f"No lat/lon found for system_id={system_id}")
            lat, lon = lat_lon_map[system_id]

        fit_result = _run_blocks_for_system(
            input_csv=input_csv,
            output_dir=output_dir,
            timestamp_col=timestamp_col,
            power_col=power_col,
            timezone_str=timezone_str,
            latitude=lat,
            longitude=lon,
        )
        status = "ok"
        error = None
    except Exception as exc:
        fit_result = {}
        status = "failed"
        error = {"type": exc.__class__.__name__, "message": str(exc)}
        (output_dir / "00_error.txt").write_text(traceback.format_exc(), encoding="utf-8")

    finished = _now_iso()
    runtime = perf_counter() - t0
    payload = {
        "system_id": system_id,
        "input_csv": str(input_csv),
        "status": status,
        "error": error,
        "started_at": started,
        "finished_at": finished,
        "runtime_seconds": runtime,
    }
    (output_dir / "00_status.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    summary = {
        **payload,
        "model_type": fit_result.get("model_type"),
        "tilt_deg": fit_result.get("tilt_deg"),
        "azimuth_deg": fit_result.get("azimuth_deg"),
        "azimuth_center_deg": fit_result.get("azimuth_center_deg"),
        "weight_east": fit_result.get("weight_east"),
        "score_rmse": fit_result.get("score_rmse"),
        "score_bic": fit_result.get("score_bic"),
    }
    return summary


def _process_one_worker(kwargs: dict[str, Any]) -> dict[str, Any]:
    return _process_one(**kwargs)


def run_batch(
    *,
    input_dir: str | Path,
    pattern: str,
    output_root: str | Path,
    timestamp_col: str = "time",
    power_col: str = "ac_power_w",
    timezone_str: str = "Etc/GMT-1",
    latitude: float | None = None,
    longitude: float | None = None,
    systems_metadata_csv: str | Path | None = None,
    system_id_col: str = "system_id",
    jobs: int = 1,
    skip_existing: bool = False,
) -> pd.DataFrame:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    inputs = discover_input_csvs(input_dir, pattern)
    lat_lon_map = _load_lat_lon_map(systems_metadata_csv, system_id_col=system_id_col)

    kwargs_list = [
        {
            "input_csv": p,
            "output_root": output_root,
            "timestamp_col": timestamp_col,
            "power_col": power_col,
            "timezone_str": timezone_str,
            "latitude": latitude,
            "longitude": longitude,
            "lat_lon_map": lat_lon_map,
            "skip_existing": skip_existing,
        }
        for p in inputs
    ]

    rows: list[dict[str, Any]] = []
    if jobs <= 1:
        for kw in kwargs_list:
            rows.append(_process_one(**kw))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(_process_one_worker, kw) for kw in kwargs_list]
            for fut in as_completed(futs):
                rows.append(fut.result())
        rows.sort(key=lambda r: _system_sort_key(Path(r["input_csv"])))

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_root / "batch_summary.csv", index=False)
    return summary_df
