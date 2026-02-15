from __future__ import annotations

import pandas as pd
import pytest

from pv_profiler.validation import INTERNAL_TZ, parse_and_validate_timestamp_index, validate_regular_sampling


def test_naive_localized_to_fixed_offset() -> None:
    idx = parse_and_validate_timestamp_index([
        "2024-01-01T00:00:00",
        "2024-01-01T00:05:00",
    ])
    assert str(idx.tz) == INTERNAL_TZ


def test_reject_europe_berlin_dst_timezone() -> None:
    berlin_idx = pd.date_range("2024-01-01 00:00:00", periods=2, freq="5min", tz="Europe/Berlin")
    with pytest.raises(ValueError, match="Europe/Berlin"):
        parse_and_validate_timestamp_index(berlin_idx)


def test_sampling_validation_strict_5min() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="10min", tz=INTERNAL_TZ)
    with pytest.raises(ValueError, match="Irregular sampling"):
        validate_regular_sampling(idx)
