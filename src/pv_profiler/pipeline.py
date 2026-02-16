from __future__ import annotations

from pathlib import Path

from .block_diagnostics import compute_diagnostics
from .block_io import load_input_for_sdt, read_metadata, read_power_timeseries, write_input_loader_artifacts
from .block_orientation import estimate_orientation
from .block_sdt import run_sdt_onboarding
from .types import InputLoaderResult, RunSingleResult


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
