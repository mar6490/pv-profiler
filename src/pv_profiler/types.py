from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


@dataclass
class TimeSeriesData:
    data: pd.Series
    timezone: str | None = None
    source_column: str = "P_AC"


@dataclass
class InputDiagnostics:
    shape: tuple[int, int]
    columns: list[str]
    min_time: str | None
    max_time: str | None
    dominant_timedelta: str | None
    sampling_summary: dict[str, int]
    share_positive_power: float
    share_nan_power: float
    share_non_null_power: float
    index_monotonic_increasing: bool
    min_power: float | None
    max_power: float | None
    decisions: list[str] = field(default_factory=list)


@dataclass
class InputLoaderResult:
    data: pd.DataFrame
    diagnostics: InputDiagnostics


@dataclass
class SdtBlockResult:
    status: str
    solver: str
    fix_shifts: bool
    report: dict[str, Any] | None = None
    raw_data_matrix: pd.DataFrame | None = None
    filled_data_matrix: pd.DataFrame | None = None
    daily_flags: pd.DataFrame | None = None
    error: dict[str, Any] | None = None
    message: str | None = None


@dataclass
class OrientationResult:
    tilt: float
    azimuth: float
    score: float
    method: str = "grid-search-clear-sky"


@dataclass
class DiagnosticsResult:
    n_samples: int
    coverage_fraction: float
    frequency: str | None
    warnings: list[str] = field(default_factory=list)


@dataclass
class RunSingleResult:
    orientation: OrientationResult
    diagnostics: DiagnosticsResult
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload
