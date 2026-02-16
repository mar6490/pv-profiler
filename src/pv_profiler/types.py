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
