"""Block A implementation using solar-data-tools (SDT)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pv_profiler.validation import INTERNAL_TZ

LOGGER = logging.getLogger(__name__)


def _sdt_dir_attrs(dh: object) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in dir(dh):
        if name.startswith("_"):
            continue
        try:
            value = getattr(dh, name)
        except Exception:  # pragma: no cover
            continue
        if isinstance(value, (int, float, str, bool, type(None))):
            out[name] = str(value)
        elif isinstance(value, (pd.Series, pd.DataFrame, np.ndarray, list, tuple, dict)):
            out[name] = type(value).__name__
    return out


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def _shape_for(value: object) -> str | None:
    if hasattr(value, "shape"):
        return str(getattr(value, "shape"))
    if isinstance(value, (list, tuple, dict)):
        return str((len(value),))
    return None


def write_sdt_introspect(
    dh: object,
    out_dir: str | Path,
    power_col: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write SDT introspection payload for debugging."""
    keywords = [
        "data_frame",
        "raw_data_matrix",
        "filled_data_matrix",
        "processed",
        "clean",
        "shift",
        "time",
        "index",
        "boolean_masks",
        "daily_flags",
    ]

    out: dict[str, Any] = {
        "power_col": power_col,
        "data_frame_columns": [],
        "data_frame_head": "",
        "selected_attributes": [],
        "report": None,
        "extra": extra or {},
    }

    try:
        df = getattr(dh, "data_frame", None)
        if isinstance(df, pd.DataFrame):
            out["data_frame_columns"] = list(df.columns)
            out["data_frame_head"] = df.head(3).to_string()
    except Exception as exc:  # pragma: no cover
        out["data_frame_head"] = f"<unavailable: {exc}>"

    selected_attributes: list[dict[str, Any]] = []
    for name in dir(dh):
        if name.startswith("_"):
            continue
        lname = name.lower()
        if not any(k in lname for k in keywords):
            continue
        try:
            value = getattr(dh, name)
            selected_attributes.append(
                {
                    "name": name,
                    "type": type(value).__name__,
                    "shape": _shape_for(value),
                }
            )
        except Exception as exc:  # pragma: no cover
            selected_attributes.append({"name": name, "type": "<error>", "shape": str(exc)})

    out["selected_attributes"] = selected_attributes

    try:
        out["report"] = dh.report(return_values=True)
    except Exception as exc:  # pragma: no cover
        out["report"] = {"error": str(exc)}

    p = Path(out_dir) / "07_sdt_introspect.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return out


def extract_clean_power_series(dh: object, power_col: str = "ac_power") -> tuple[pd.Series, str]:
    """Extract SDT shift-corrected clean power series without silent fallback."""
    candidates = [
        "clean_power",
        "clean_power_series",
        "power_clean",
        "power_signals_d",
        "filled_data_matrix",
    ]

    for attr in candidates:
        if not hasattr(dh, attr):
            continue
        raw = getattr(dh, attr)
        if isinstance(raw, pd.Series):
            series = raw.copy()
        elif isinstance(raw, pd.DataFrame) and power_col in raw.columns:
            series = raw[power_col].copy()
        else:
            continue

        if not isinstance(series.index, pd.DatetimeIndex):
            continue
        if series.index.tz is None:
            series.index = series.index.tz_localize(INTERNAL_TZ)
        else:
            series.index = series.index.tz_convert(INTERNAL_TZ)
        return series.rename("ac_power_clean"), f"dh.{attr}"

    raise RuntimeError(
        "Unable to identify SDT cleaned shift-corrected power series through public attributes. "
        "See 07_sdt_introspect.json for available attributes."
    )


def _clipping_mask_heuristic(ac_power_clean: pd.Series, lat: float, lon: float) -> pd.Series:
    import pvlib

    solar_pos = pvlib.solarposition.get_solarposition(ac_power_clean.index, latitude=lat, longitude=lon)
    daytime = solar_pos["apparent_elevation"] > 3.0
    day_max = ac_power_clean.where(daytime).groupby(ac_power_clean.index.date).transform("max")
    threshold = 0.98 * day_max
    clipped = daytime & (ac_power_clean >= threshold)
    clipped.name = "is_clipped_time"
    return clipped.fillna(False)


def run_block_a(
    ac_power: pd.Series,
    lat: float,
    lon: float,
    power_col: str = "ac_power",
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run Block A (A1/A2/A3) and return artifacts and summaries."""
    from solardatatools import DataHandler

    if not isinstance(ac_power.index, pd.DatetimeIndex):
        raise ValueError("ac_power index must be DatetimeIndex.")

    ac_power = ac_power.copy()
    if str(ac_power.index.tz) != INTERNAL_TZ:
        ac_power.index = ac_power.index.tz_convert(INTERNAL_TZ)

    parsed = ac_power.rename("ac_power").to_frame()

    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        _write_parquet(parsed, out_path / "01_parsed_tzaware.parquet")

    dh = DataHandler(parsed)
    dh.run_pipeline(power_col=power_col, fix_shifts=True, verbose=False)

    report = None
    if out_dir is not None:
        write_sdt_introspect(dh, out_dir=out_dir, power_col=power_col, extra={"stage": "after_run_pipeline"})

    clear_col = pd.Series(False, index=parsed.index, name="is_clear_time")
    if hasattr(dh, "boolean_masks") and hasattr(dh.boolean_masks, "clear_times"):
        dh.augment_data_frame(dh.boolean_masks.clear_times, "is_clear_time")
    if hasattr(dh, "data_frame") and isinstance(dh.data_frame, pd.DataFrame) and "is_clear_time" in dh.data_frame.columns:
        clear_col = dh.data_frame["is_clear_time"].reindex(parsed.index).fillna(False).astype(bool)
        clear_col.name = "is_clear_time"

    if out_dir is not None:
        _write_parquet(clear_col.to_frame(), Path(out_dir) / "03_clear_times_mask.parquet")

    try:
        ac_power_clean, clean_source = extract_clean_power_series(dh, power_col=power_col)
        clean_error = None
    except Exception as exc:
        clean_source = "unresolved"
        clean_error = str(exc)
        if out_dir is not None:
            write_sdt_introspect(
                dh,
                out_dir=out_dir,
                power_col=power_col,
                extra={"stage": "extract_clean_power_series", "error": str(exc)},
            )
        raise
    finally:
        try:
            report = dh.report(return_values=True)
        except Exception as exc:  # pragma: no cover
            report = {"error": str(exc)}

    clear_col = clear_col.reindex(ac_power_clean.index).fillna(False).astype(bool)
    clear_col.name = "is_clear_time"

    clipped = _clipping_mask_heuristic(ac_power_clean, lat=lat, lon=lon)
    clipped = clipped.reindex(ac_power_clean.index).fillna(False)

    local_dates = pd.Index(ac_power_clean.index.tz_convert(INTERNAL_TZ).date, name="date")
    clipping_day_share = (
        pd.DataFrame({"date": local_dates, "is_clipped": clipped.astype(int)})
        .groupby("date", as_index=False)["is_clipped"]
        .mean()
        .rename(columns={"is_clipped": "clipping_day_share"})
    )

    clipping_summary = {
        "clipping_day_share_mean": float(clipping_day_share["clipping_day_share"].mean()),
        "clipping_fraction_day_median": float(clipping_day_share["clipping_day_share"].median()),
        "n_days": int(clipping_day_share.shape[0]),
    }

    n_days_total = int(len(pd.unique(local_dates)))
    n_clear_times = int(clear_col.sum())
    clear_frac = float(n_clear_times / max(len(clear_col), 1))

    sdt_summary = {
        "n_days_total": n_days_total,
        "n_clear_times": n_clear_times,
        "clear_time_fraction_overall": clear_frac,
        "time_shift_correction_applied": report.get("time shift correction") if isinstance(report, dict) else None,
        "time_zone_correction": report.get("time zone correction") if isinstance(report, dict) else None,
    }

    daily_flags = pd.DataFrame(
        {
            "date": clipping_day_share["date"].astype(str),
            "clipping_day_share": clipping_day_share["clipping_day_share"],
        }
    )

    introspect = {
        "data_frame_columns": list(getattr(dh, "data_frame", pd.DataFrame()).columns),
        "dh_attributes": _sdt_dir_attrs(dh),
        "report": report if isinstance(report, dict) else {"raw": str(report)},
        "clean_power_source": clean_source,
        "clean_power_error": clean_error,
    }

    return {
        "parsed": parsed,
        "ac_power_clean": ac_power_clean.to_frame(),
        "clear_times": clear_col.to_frame(),
        "clipped_times": clipped.to_frame(),
        "daily_flags": daily_flags,
        "clipping_summary": clipping_summary,
        "sdt_summary": sdt_summary,
        "sdt_introspect": introspect,
    }


def apply_exclusion_rules(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    """Apply A3 exclusion logic."""
    pconf = config.get("pipeline", {})
    clip_share_thr = float(pconf.get("clipping_threshold_day_share", 0.10))
    clip_median_thr = float(pconf.get("clipping_fraction_day_median", 0.01))
    clear_frac_min = float(pconf.get("clear_time_fraction_min", 0.005))

    clipping_day_share_mean = float(summary.get("clipping_day_share_mean", 0.0))
    clipping_fraction_day_median = float(summary.get("clipping_fraction_day_median", 0.0))
    clear_time_fraction = float(summary.get("clear_time_fraction_overall", 0.0))

    time_shift_applied = summary.get("time_shift_correction_applied")
    tz_corr = summary.get("time_zone_correction")

    suspect_large_shift = False
    if bool(time_shift_applied) and isinstance(tz_corr, (int, float)):
        suspect_large_shift = abs(float(tz_corr)) >= 2.0

    return {
        "exclude_clipping": clipping_day_share_mean >= clip_share_thr or clipping_fraction_day_median >= clip_median_thr,
        "exclude_low_clear": clear_time_fraction < clear_frac_min,
        "suspect_large_shift": suspect_large_shift,
    }
