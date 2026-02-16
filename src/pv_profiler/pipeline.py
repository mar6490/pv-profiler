from __future__ import annotations

from pathlib import Path

from .block_diagnostics import compute_diagnostics
from .block_io import read_metadata, read_power_timeseries
from .block_orientation import estimate_orientation
from .block_sdt import run_sdt_onboarding
from .types import RunSingleResult


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
