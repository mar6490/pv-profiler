"""Batch orchestration for run-single/run-wide/run commands."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from pv_profiler.io import (
    read_manifest,
    read_plants_metadata,
    read_single_plant,
    read_wide_plants,
    write_csv,
    write_json,
    write_parquet,
)
from pv_profiler.normalization import compute_daily_peak_and_norm, compute_fit_mask
from pv_profiler.sdt_pipeline import apply_exclusion_rules, run_block_a
from pv_profiler.utils import ensure_dir, utc_timestamp_label

LOGGER = logging.getLogger(__name__)


def _lookup_location(system_id: str, metadata: pd.DataFrame, lat: float | None, lon: float | None) -> tuple[float, float]:
    row = metadata.loc[metadata["system_id"].astype(str) == str(system_id)]
    if not row.empty:
        return float(row.iloc[0]["lat"]), float(row.iloc[0]["lon"])
    if lat is not None and lon is not None:
        return float(lat), float(lon)
    raise ValueError(f"No location for system_id={system_id}. Provide --lat/--lon or add metadata row.")


def _system_output_dir(output_root: str | Path, system_id: str, run_label: str) -> Path:
    return ensure_dir(Path(output_root) / str(system_id) / run_label)


def process_single_system(
    system_id: str,
    df: pd.DataFrame,
    config: dict[str, Any],
    lat: float,
    lon: float,
    run_label: str,
) -> dict[str, Any]:
    """Run full A-C pipeline for one system and write artifacts."""
    out_dir = _system_output_dir(config["paths"]["output_root"], system_id=system_id, run_label=run_label)
    power = df["ac_power"].rename("ac_power")

    block_a = run_block_a(power, lat=lat, lon=lon)

    write_parquet(block_a["parsed"], out_dir / "01_parsed_tzaware.parquet")
    write_parquet(block_a["ac_power_clean"], out_dir / "02_cleaned_timeshift_fixed.parquet")
    write_parquet(block_a["clear_times"], out_dir / "03_clear_times_mask.parquet")
    write_csv(block_a["daily_flags"], out_dir / "04_daily_flags.csv")
    write_json(block_a["clipping_summary"], out_dir / "05_clipping_summary.json")
    write_parquet(block_a["clipped_times"], out_dir / "06_clipped_times_mask.parquet")
    write_json(block_a["sdt_summary"], out_dir / "07_sdt_summary.json")
    write_json(block_a["sdt_introspect"], out_dir / "07_sdt_introspect.json")

    clip_summary = block_a["clipping_summary"]
    sdt_summary = block_a["sdt_summary"]

    daily_peak, p_norm = compute_daily_peak_and_norm(
        ac_power_clean=block_a["ac_power_clean"]["ac_power_clean"],
        is_clipped_time=block_a["clipped_times"]["is_clipped_time"],
        lat=lat,
        lon=lon,
    )
    write_csv(daily_peak, out_dir / "08_daily_peak.csv")
    write_parquet(p_norm.to_frame(), out_dir / "09_p_norm.parquet")

    tau = float(config.get("pipeline", {}).get("fit_tau", 0.03))
    fit_mask, daily_fit_fraction = compute_fit_mask(
        p_norm=p_norm,
        is_clear_time=block_a["clear_times"]["is_clear_time"],
        is_clipped_time=block_a["clipped_times"]["is_clipped_time"],
        tau=tau,
    )
    write_parquet(fit_mask.to_frame(), out_dir / "11_fit_mask.parquet")
    write_csv(daily_fit_fraction, out_dir / "12_daily_fit_fraction.csv")

    merged_summary = {
        "system_id": system_id,
        "run_label": run_label,
        "lat": lat,
        "lon": lon,
        **clip_summary,
        **sdt_summary,
        "fit_tau": tau,
        "fit_fraction_mean": float(daily_fit_fraction["daily_fit_fraction"].mean()) if not daily_fit_fraction.empty else 0.0,
    }
    merged_summary.update(apply_exclusion_rules(merged_summary, config=config))

    write_json(merged_summary, out_dir / "summary.json")
    LOGGER.info("Finished system_id=%s at %s", system_id, out_dir)
    return merged_summary


def run_single(
    *,
    system_id: str,
    input_path: str,
    config: dict[str, Any],
    lat: float | None,
    lon: float | None,
) -> dict[str, Any]:
    metadata = read_plants_metadata(config["paths"]["plants_csv"])
    df = read_single_plant(input_path)
    final_lat, final_lon = _lookup_location(system_id, metadata, lat, lon)
    run_label = utc_timestamp_label()
    return process_single_system(system_id, df, config, final_lat, final_lon, run_label)


def run_wide(*, input_path: str, config: dict[str, Any], system_ids: list[str] | None = None) -> list[dict[str, Any]]:
    metadata = read_plants_metadata(config["paths"]["plants_csv"])
    wide = read_wide_plants(input_path)
    run_label = utc_timestamp_label()

    selected_columns = system_ids if system_ids else list(wide.columns)
    results = []
    for system_id in selected_columns:
        if system_id not in wide.columns:
            raise ValueError(f"system_id={system_id} not found in wide input columns.")
        single_df = pd.DataFrame({"ac_power": wide[system_id]}, index=wide.index)
        lat, lon = _lookup_location(system_id, metadata, None, None)
        results.append(process_single_system(system_id, single_df, config, lat, lon, run_label))
    return results


def run_manifest(*, manifest_path: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = read_plants_metadata(config["paths"]["plants_csv"])
    manifest = read_manifest(manifest_path)
    run_label = utc_timestamp_label()
    results: list[dict[str, Any]] = []

    for row in manifest.to_dict(orient="records"):
        system_id = str(row["system_id"])
        if "path" not in row or pd.isna(row["path"]):
            raise ValueError("Manifest rows must include 'path' for run mode.")
        df = read_single_plant(str(row["path"]))
        lat = float(row["lat"]) if "lat" in row and pd.notna(row["lat"]) else None
        lon = float(row["lon"]) if "lon" in row and pd.notna(row["lon"]) else None
        final_lat, final_lon = _lookup_location(system_id, metadata, lat, lon)
        results.append(process_single_system(system_id, df, config, final_lat, final_lon, run_label))

    return results
