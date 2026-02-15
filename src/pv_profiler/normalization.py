"""Blocks B and C: normalization and fit mask generation."""

from __future__ import annotations

import pandas as pd

from pv_profiler.validation import INTERNAL_TZ


def daytime_mask(index: pd.DatetimeIndex, lat: float, lon: float) -> pd.Series:
    """Return daytime mask using apparent elevation > 3° in fixed internal timezone."""
    if str(index.tz) != INTERNAL_TZ:
        index = index.tz_convert(INTERNAL_TZ)
    import pvlib

    solar_pos = pvlib.solarposition.get_solarposition(index, latitude=lat, longitude=lon)
    return solar_pos["apparent_elevation"] > 3.0


def compute_daily_peak_and_norm(
    ac_power_clean: pd.Series,
    is_clipped_time: pd.Series,
    lat: float,
    lon: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """Compute Block B daily peak summary and normalized power series."""
    day_mask = daytime_mask(ac_power_clean.index, lat=lat, lon=lon)
    valid = day_mask & (~is_clipped_time.fillna(False))
    selected = ac_power_clean.where(valid)
    p_peak_day = float(selected.quantile(0.995))
    if p_peak_day <= 0 or pd.isna(p_peak_day):
        raise ValueError("P_peak_day is invalid; cannot normalize series.")

    p_norm = ac_power_clean / p_peak_day
    p_norm.name = "p_norm"
    daily = pd.DataFrame(
        {
            "metric": ["p_peak_day_q995"],
            "value": [p_peak_day],
        }
    )
    return daily, p_norm


def compute_fit_mask(
    p_norm: pd.Series,
    is_clear_time: pd.Series,
    is_clipped_time: pd.Series,
    tau: float,
) -> tuple[pd.Series, pd.DataFrame]:
    """Compute Block C fit mask and daily fraction summary."""
    delta = (p_norm - p_norm.shift(1)).abs()
    is_smooth = delta <= tau
    fit_mask = (is_clear_time.fillna(False)) & (is_smooth.fillna(False)) & (~is_clipped_time.fillna(False))
    fit_mask.name = "fit_mask"

    local_date = p_norm.index.tz_convert(INTERNAL_TZ).date
    daily_fraction = (
        pd.DataFrame({"date": local_date, "fit_mask": fit_mask.astype(int)})
        .groupby("date", as_index=False)["fit_mask"]
        .mean()
        .rename(columns={"fit_mask": "daily_fit_fraction"})
    )
    return fit_mask, daily_fraction
