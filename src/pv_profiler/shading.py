"""Block E: shading map and metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pv_profiler.validation import INTERNAL_TZ


def _bin_centers(values: pd.Series, bin_deg: int, min_edge: int, max_edge: int) -> tuple[pd.Series, pd.Series]:
    bins = np.arange(min_edge, max_edge + bin_deg, bin_deg)
    labels = bins[:-1] + (bin_deg / 2.0)
    cuts = pd.cut(values, bins=bins, include_lowest=True, right=False, labels=labels)
    return cuts.astype(float), pd.Series(labels)


def compute_shading(
    ac_power_clean: pd.Series,
    fit_times: pd.Series,
    poa_cs: pd.Series,
    kWp_effective: float,
    *,
    lat: float,
    lon: float,
    config: dict[str, Any],
    plot_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute shading map parquet payload and summary metrics."""
    import matplotlib.pyplot as plt
    import pvlib

    cfg = config.get("shading", {})
    r_max = float(cfg.get("r_max", 1.2))
    az_bin = int(cfg.get("az_bin_deg", 5))
    el_bin = int(cfg.get("el_bin_deg", 5))
    morning_lo, morning_hi = cfg.get("morning_sector_deg", [60, 150])
    evening_lo, evening_hi = cfg.get("evening_sector_deg", [210, 300])

    times = ac_power_clean.index
    location = pvlib.location.Location(latitude=lat, longitude=lon, tz=INTERNAL_TZ)
    solar = location.get_solarposition(times)

    p_hat_unshaded_scaled = kWp_effective * (poa_cs / 1000.0)
    r = (ac_power_clean / p_hat_unshaded_scaled).rename("r")

    fit_mask = fit_times.reindex(times).fillna(False).astype(bool)
    poa_filter = poa_cs.reindex(times) > 200.0
    residual_base = fit_mask & poa_filter & r.notna() & (r > 0)

    az = solar["azimuth"].rename("azimuth")
    el = solar["apparent_elevation"].rename("elevation")

    residual_df = pd.DataFrame({"azimuth": az, "elevation": el, "r": r})[residual_base]
    az_bin_center, _ = _bin_centers(residual_df["azimuth"], az_bin, 0, 360)
    el_bin_center, _ = _bin_centers(residual_df["elevation"], el_bin, 0, 90)
    residual_df = residual_df.assign(az_bin_center=az_bin_center, el_bin_center=el_bin_center).dropna(
        subset=["az_bin_center", "el_bin_center"]
    )

    shading_map = (
        residual_df.groupby(["az_bin_center", "el_bin_center"], as_index=False)
        .agg(median_r=("r", "median"), count=("r", "size"))
        .sort_values(["az_bin_center", "el_bin_center"])
    )

    idx_filter = residual_base & (poa_cs > 600.0) & (el > 20.0) & (r <= r_max)
    r_for_idx = r[idx_filter]

    global_r = float(r_for_idx.median()) if not r_for_idx.empty else np.nan
    morning_mask = idx_filter & (az >= float(morning_lo)) & (az <= float(morning_hi))
    evening_mask = idx_filter & (az >= float(evening_lo)) & (az <= float(evening_hi))

    r_morning = float(r[morning_mask].median()) if morning_mask.any() else np.nan
    r_evening = float(r[evening_mask].median()) if evening_mask.any() else np.nan

    metrics = {
        "r_max": r_max,
        "poa_residual_threshold_wm2": 200.0,
        "poa_index_threshold_wm2": 600.0,
        "elevation_threshold_deg": 20.0,
        "global_shading_index": float(1.0 - global_r) if np.isfinite(global_r) else None,
        "morning_shading_index": float(1.0 - r_morning) if np.isfinite(r_morning) else None,
        "evening_shading_index": float(1.0 - r_evening) if np.isfinite(r_evening) else None,
        "count_residual_points": int(residual_base.sum()),
        "count_index_points": int(idx_filter.sum()),
        "count_morning_points": int(morning_mask.sum()),
        "count_evening_points": int(evening_mask.sum()),
        "morning_sector_deg": [float(morning_lo), float(morning_hi)],
        "evening_sector_deg": [float(evening_lo), float(evening_hi)],
    }

    heat = shading_map.pivot(index="el_bin_center", columns="az_bin_center", values="median_r")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(
        heat.sort_index().values,
        aspect="auto",
        origin="lower",
        extent=[
            float(heat.columns.min()) if not heat.empty else 0,
            float(heat.columns.max()) if not heat.empty else 360,
            float(heat.index.min()) if not heat.empty else 0,
            float(heat.index.max()) if not heat.empty else 90,
        ],
    )
    ax.set_xlabel("Solar azimuth [deg]")
    ax.set_ylabel("Solar elevation [deg]")
    ax.set_title("Shading map (median residual r)")
    fig.colorbar(im, ax=ax, label="median r")
    path = Path(plot_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return shading_map, metrics
