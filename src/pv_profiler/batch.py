"""Batch orchestration for run-single/run-wide/run commands."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from pv_profiler.capacity import estimate_kWp_effective
from pv_profiler.io import (
    read_manifest,
    read_metadata_json,
    read_plants_metadata,
    read_single_plant,
    read_wide_plants,
    write_csv,
    write_json,
    write_parquet,
)
from pv_profiler.normalization import compute_daily_peak_and_norm, compute_fit_mask
from pv_profiler.orientation import fit_orientation
from pv_profiler.sdt_pipeline import apply_exclusion_rules, run_block_a
from pv_profiler.shading import compute_shading
from pv_profiler.utils import ensure_dir, utc_timestamp_label
from pv_profiler.validation import INTERNAL_TZ

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


def _load_single_for_run_single(input_path: str, config: dict[str, Any]) -> pd.DataFrame:
    input_cfg = config.get("input", {})
    timestamp_col = str(input_cfg.get("timestamp_col", "timestamp"))
    power_col = str(input_cfg.get("power_col", "ac_power"))
    sep = str(input_cfg.get("sep", ","))
    encoding = str(input_cfg.get("encoding", "utf-8-sig"))
    tz_handling = str(input_cfg.get("tz_handling", "naive")).lower()

    df = pd.read_csv(
        input_path,
        sep=sep,
        encoding=encoding,
        usecols=[timestamp_col, power_col],
        low_memory=False,
    )
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col]).set_index(timestamp_col).sort_index()

    if tz_handling == "naive":
        if df.index.tz is None:
            df.index = df.index.tz_localize(INTERNAL_TZ)
        else:
            df.index = df.index.tz_convert(INTERNAL_TZ)

    power_df = df[[power_col]].rename(columns={power_col: "ac_power"})
    power_df["ac_power"] = pd.to_numeric(power_df["ac_power"], errors="coerce").fillna(0).clip(lower=0)
    return power_df


def process_single_system(
    system_id: str,
    df: pd.DataFrame,
    config: dict[str, Any],
    lat: float,
    lon: float,
    run_label: str,
    sdt_power_col: str = "ac_power",
) -> dict[str, Any]:
    """Run full A-E pipeline for one system and write artifacts."""
    out_dir = _system_output_dir(config["paths"]["output_root"], system_id=system_id, run_label=run_label)
    power = df["ac_power"].rename("ac_power")

    block_a = run_block_a(power, lat=lat, lon=lon, power_col=sdt_power_col, config=config, out_dir=out_dir)

    write_parquet(block_a["parsed"], out_dir / "01_parsed_tzaware.parquet")
    write_parquet(block_a["ac_power_clean"], out_dir / "02_cleaned_timeshift_fixed.parquet")
    write_parquet(block_a["fit_times"], out_dir / "03_fit_times_mask.parquet")
    write_csv(block_a["daily_flags"], out_dir / "04_daily_flags.csv")
    write_json(block_a["clipping_summary"], out_dir / "05_clipping_summary.json")
    write_parquet(block_a["clipped_times"], out_dir / "06_clipped_times_mask.parquet")
    write_json(block_a["sdt_summary"], out_dir / "07_sdt_summary.json")
    write_json(block_a["sdt_introspect"], out_dir / "07_sdt_introspect.json")
    if "raw_data_matrix" in block_a:
        write_parquet(block_a["raw_data_matrix"], out_dir / "sdt_raw_data_matrix.parquet")
    if "filled_data_matrix" in block_a:
        write_parquet(block_a["filled_data_matrix"], out_dir / "sdt_filled_data_matrix.parquet")
    if "sdt_daily_flags" in block_a:
        write_csv(block_a["sdt_daily_flags"], out_dir / "sdt_daily_flags.csv")
    if "filled_timeseries" in block_a:
        write_parquet(block_a["filled_timeseries"], out_dir / "sdt_filled_timeseries.parquet")
        write_csv(block_a["filled_timeseries"].reset_index(), out_dir / "sdt_filled_timeseries.csv")

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
        is_clear_time=block_a["fit_times"]["is_fit_time"],
        is_clipped_time=block_a["clipped_times"]["is_clipped_time"],
        tau=tau,
    )
    write_parquet(fit_mask.to_frame(), out_dir / "11_fit_mask.parquet")
    write_csv(daily_fit_fraction, out_dir / "12_daily_fit_fraction.csv")

    merged_summary: dict[str, Any] = {
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

    excluded = bool(merged_summary.get("exclude_clipping") or merged_summary.get("exclude_low_clear"))
    if excluded:
        merged_summary["pipeline_de_skipped"] = True
        write_json(merged_summary, out_dir / "summary.json")
        LOGGER.info("Skipped D/E for excluded system_id=%s", system_id)
        return merged_summary

    orientation_artifacts = fit_orientation(
        ac_power_clean=block_a["ac_power_clean"]["ac_power_clean"],
        fit_mask=fit_mask,
        lat=lat,
        lon=lon,
        config=config,
    )

    capacity_result = estimate_kWp_effective(
        ac_power_clean=block_a["ac_power_clean"]["ac_power_clean"],
        poa_cs=orientation_artifacts.poa_unshaded,
        fit_mask=fit_mask,
        poa_threshold_wm2=float(config.get("capacity", {}).get("poa_threshold_wm2", 600.0)),
    )

    orientation_result = dict(orientation_artifacts.result)
    orientation_result.update(capacity_result)
    write_json(orientation_result, out_dir / "13_orientation_result.json")
    write_csv(orientation_artifacts.diagnostics, out_dir / "14_fit_diagnostics.csv")

    shading_map, shading_metrics = compute_shading(
        ac_power_clean=block_a["ac_power_clean"]["ac_power_clean"],
        fit_times=block_a["fit_times"]["is_fit_time"],
        poa_cs=orientation_artifacts.poa_unshaded,
        kWp_effective=capacity_result["kWp_effective"],
        lat=lat,
        lon=lon,
        config=config,
        plot_path=out_dir / "shading_map.png",
    )
    write_parquet(shading_map, out_dir / "shading_map.parquet")
    write_json(shading_metrics, out_dir / "shading_metrics.json")

    merged_summary.update(
        {
            "pipeline_de_skipped": False,
            "orientation_model_type": orientation_result.get("model_type"),
            "orientation_tilt_deg": orientation_result.get("tilt_deg"),
            "orientation_score_rmse": orientation_result.get("score_rmse"),
            "kWp_effective": orientation_result.get("kWp_effective"),
            "global_shading_index": shading_metrics.get("global_shading_index"),
            "morning_shading_index": shading_metrics.get("morning_shading_index"),
            "evening_shading_index": shading_metrics.get("evening_shading_index"),
        }
    )

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
    metadata_json: str | None = None,
    output_root: str | None = None,
) -> dict[str, Any]:
    if output_root:
        config = {**config, "paths": {**config["paths"], "output_root": output_root}}

    metadata = read_plants_metadata(config["paths"]["plants_csv"])
    df = _load_single_for_run_single(input_path, config)

    if metadata_json:
        m = read_metadata_json(metadata_json)
        final_lat, final_lon = float(m["lat"]), float(m["lon"])
    else:
        final_lat, final_lon = _lookup_location(system_id, metadata, lat, lon)

    run_label = utc_timestamp_label()
    return process_single_system(system_id, df, config, final_lat, final_lon, run_label, sdt_power_col="power")


def run_wide(*, input_path: str, config: dict[str, Any], system_ids: list[str] | None = None) -> list[dict[str, Any]]:
    metadata = read_plants_metadata(config["paths"]["plants_csv"])
    timestamp_col = str(config.get("input", {}).get("timestamp_col", "timestamp"))
    wide = read_wide_plants(input_path, timestamp_col=timestamp_col)
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
    timestamp_col = str(config.get("input", {}).get("timestamp_col", "timestamp"))
    power_col = str(config.get("input", {}).get("power_col", "ac_power"))

    manifest = read_manifest(manifest_path)
    run_label = utc_timestamp_label()
    results: list[dict[str, Any]] = []

    for row in manifest.to_dict(orient="records"):
        system_id = str(row["system_id"])
        if "path" not in row or pd.isna(row["path"]):
            raise ValueError("Manifest rows must include 'path' for run mode.")
        df = read_single_plant(str(row["path"]), timestamp_col=timestamp_col, power_col=power_col)
        lat = float(row["lat"]) if "lat" in row and pd.notna(row["lat"]) else None
        lon = float(row["lon"]) if "lon" in row and pd.notna(row["lon"]) else None
        final_lat, final_lon = _lookup_location(system_id, metadata, lat, lon)
        results.append(process_single_system(system_id, df, config, final_lat, final_lon, run_label))

    return results
