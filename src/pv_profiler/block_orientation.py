from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib

from .types import OrientationResult


def estimate_orientation(
    power_series: pd.Series,
    latitude: float,
    longitude: float,
    altitude: float | None = None,
) -> OrientationResult:
    if power_series.index.tz is None:
        raise ValueError("Power series index must be timezone-aware for orientation estimation.")

    location = pvlib.location.Location(latitude=latitude, longitude=longitude, altitude=altitude)

    clearsky = location.get_clearsky(power_series.index)
    solar_position = location.get_solarposition(power_series.index)

    target = power_series.clip(lower=0)
    target_norm = target / (target.max() or 1.0)

    best = OrientationResult(tilt=30.0, azimuth=180.0, score=float("-inf"))
    tilts = np.arange(5, 61, 1)
    azimuths = np.arange(90, 271, 2)

    ghi = clearsky["ghi"]
    dni = clearsky["dni"]
    dhi = clearsky["dhi"]

    for tilt in tilts:
        for azimuth in azimuths:
            poa = pvlib.irradiance.get_total_irradiance(
                surface_tilt=float(tilt),
                surface_azimuth=float(azimuth),
                solar_zenith=solar_position["apparent_zenith"],
                solar_azimuth=solar_position["azimuth"],
                dni=dni,
                ghi=ghi,
                dhi=dhi,
                model="haydavies",
            )["poa_global"].clip(lower=0)
            poa_norm = poa / (poa.max() or 1.0)
            score = float(target_norm.corr(poa_norm))
            if np.isfinite(score) and score > best.score:
                best = OrientationResult(tilt=float(tilt), azimuth=float(azimuth), score=score)

    return best
