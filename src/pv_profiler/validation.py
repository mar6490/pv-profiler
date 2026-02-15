"""Input validation for PV profiler."""

from __future__ import annotations

from datetime import timedelta
from typing import Iterable

import pandas as pd

INTERNAL_TZ = "Etc/GMT-1"
EXPECTED_FREQ = pd.Timedelta(minutes=5)


def parse_and_validate_timestamp_index(values: Iterable[object] | pd.Series) -> pd.DatetimeIndex:
    """Parse timestamps and normalize them to the fixed internal timezone.

    Rules:
    - tz-naive values are localized to ``Etc/GMT-1``.
    - tz-aware values must be fixed offset +01:00 (no DST-aware zones such as Europe/Berlin).
    """
    idx = pd.to_datetime(pd.Index(values), errors="raise")
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("Timestamp parsing did not return a DatetimeIndex.")

    if idx.tz is None:
        idx = idx.tz_localize(INTERNAL_TZ)
    else:
        tz_name = str(idx.tz)
        if "Europe/Berlin" in tz_name:
            raise ValueError("Timezone Europe/Berlin is not allowed. Use fixed +01:00 only.")
        offsets = idx.map(lambda ts: ts.utcoffset())
        allowed = timedelta(hours=1)
        if any(off != allowed for off in offsets):
            raise ValueError("Only fixed offset +01:00 timestamps are allowed.")
        idx = idx.tz_convert(INTERNAL_TZ)

    return idx


def validate_regular_sampling(index: pd.DatetimeIndex, expected_delta: pd.Timedelta = EXPECTED_FREQ) -> None:
    """Validate strict 5-minute regular sampling with no gaps or duplicates."""
    if index.has_duplicates:
        raise ValueError("Duplicate timestamps found; strict 5-minute sampling is required.")
    if len(index) < 2:
        return

    diffs = index.to_series().diff().iloc[1:]
    irregular = diffs != expected_delta
    if irregular.any():
        bad = diffs[irregular].head(5).astype(str).tolist()
        raise ValueError(
            f"Irregular sampling detected; expected 5-minute spacing without gaps. Examples: {bad}"
        )


def validate_power_series(df: pd.DataFrame, power_col: str = "ac_power") -> None:
    """Validate required power column presence and dtype compatibility."""
    if power_col not in df.columns:
        raise ValueError(f"Required column '{power_col}' is missing.")
    if not pd.api.types.is_numeric_dtype(df[power_col]):
        raise ValueError(f"Column '{power_col}' must be numeric.")
