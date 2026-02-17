from __future__ import annotations

from pathlib import Path

import pandas as pd

from .block_diagnostics import compute_diagnostics
from .block_fit import load_daily_flags, run_block3_fit_selection, write_block3_artifacts
from .block_io import load_input_for_sdt, read_metadata, read_power_timeseries, write_input_loader_artifacts
from .block_normalization import run_block4_normalize_from_parquet
from .block_orientation import estimate_orientation
from .block_sdt import run_block2_sdt, run_sdt_onboarding, write_block2_artifacts
from .types import Block3Result, InputLoaderResult, NormalizationResult, RunSingleResult, SdtBlockResult


def run_block1_input_loader(
    input_csv: str | Path,
    output_dir: str | Path,
    timestamp_col: str = "timestamp",
    power_col: str = "P_AC",
    timezone: str | None = None,
    resample_if_irregular: bool = True,
    min_samples: int = 288,
    clip_negative_power: bool = True,
) -> InputLoaderResult:
    result = load_input_for_sdt(
        input_path=input_csv,
        timestamp_col=timestamp_col,
        power_col=power_col,
        timezone=timezone,
        resample_if_irregular=resample_if_irregular,
        min_samples=min_samples,
        clip_negative_power=clip_negative_power,
    )
    write_input_loader_artifacts(result, output_dir=output_dir)
    return result


def run_single(
    input_csv: str | Path,
    metadata_path: str | Path | None = None,
    power_column: str = "P_AC",
) -> RunSingleResult:
    io_data = read_power_timeseries(input_csv, power_column=power_column)
    metadata = read_metadata(metadata_path)

    latitude = float(metadata.get("lat", 0.0))
    longitude = float(metadata.get("lon", 0.0))
    if latitude == 0.0 and longitude == 0.0:
        raise ValueError("Metadata must include valid lat/lon for orientation estimation.")

    preprocessed, sdt_warnings = run_sdt_onboarding(io_data.data)

    orientation = estimate_orientation(
        preprocessed,
        latitude=latitude,
        longitude=longitude,
        altitude=metadata.get("alt"),
    )

    diagnostics = compute_diagnostics(preprocessed, extra_warnings=sdt_warnings)
    return RunSingleResult(orientation=orientation, diagnostics=diagnostics, metadata=metadata)


def run_block2_sdt_from_df(
    power_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    solver: str = "CLARABEL",
    fix_shifts: bool = True,
    power_col: str = "power",
) -> SdtBlockResult:
    result = run_block2_sdt(
        power_df=power_df,
        solver=solver,
        fix_shifts=fix_shifts,
        power_col=power_col,
    )
    write_block2_artifacts(result, output_dir)
    return result


def run_block2_sdt_from_parquet(
    input_parquet: str | Path,
    output_dir: str | Path,
    *,
    solver: str = "CLARABEL",
    fix_shifts: bool = True,
    power_col: str = "power",
) -> SdtBlockResult:
    power_df = pd.read_parquet(input_parquet)
    if power_col not in power_df.columns and "power" in power_df.columns:
        power_col = "power"
    return run_block2_sdt_from_df(
        power_df=power_df,
        output_dir=output_dir,
        solver=solver,
        fix_shifts=fix_shifts,
        power_col=power_col,
    )


def run_block2_sdt_from_csv(
    input_csv: str | Path,
    output_dir: str | Path,
    *,
    timestamp_col: str = "timestamp",
    power_col: str = "P_AC",
    timezone: str | None = None,
    resample_if_irregular: bool = True,
    min_samples: int = 288,
    clip_negative_power: bool = True,
    solver: str = "CLARABEL",
    fix_shifts: bool = True,
) -> SdtBlockResult:
    block1 = run_block1_input_loader(
        input_csv=input_csv,
        output_dir=output_dir,
        timestamp_col=timestamp_col,
        power_col=power_col,
        timezone=timezone,
        resample_if_irregular=resample_if_irregular,
        min_samples=min_samples,
        clip_negative_power=clip_negative_power,
    )
    return run_block2_sdt_from_df(
        power_df=block1.data,
        output_dir=output_dir,
        solver=solver,
        fix_shifts=fix_shifts,
        power_col="power",
    )


def run_block3_from_files(
    input_power_parquet: str | Path,
    input_daily_flags_csv: str | Path,
    output_dir: str | Path,
    *,
    fit_mode: str = "mask_to_nan",
    min_fit_days: int = 10,
) -> Block3Result:
    power_df = pd.read_parquet(input_power_parquet)
    flags_df = load_daily_flags(input_daily_flags_csv)
    result = run_block3_fit_selection(
        power_df=power_df,
        daily_flags_df=flags_df,
        fit_mode=fit_mode,
        min_fit_days=min_fit_days,
    )
    write_block3_artifacts(result, output_dir=output_dir)
    return result


def run_block4_from_files(
    input_power_fit_parquet: str | Path,
    output_dir: str | Path,
    *,
    quantile: float = 0.995,
    min_fit_samples_day: int = 1,
    dropna_output: bool = True,
) -> NormalizationResult:
    return run_block4_normalize_from_parquet(
        input_power_fit_parquet=input_power_fit_parquet,
        output_dir=output_dir,
        quantile=quantile,
        min_fit_samples_day=min_fit_samples_day,
        dropna_output=dropna_output,
    )
