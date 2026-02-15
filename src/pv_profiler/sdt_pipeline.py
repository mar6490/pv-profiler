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
        "clean_power_source": (extra or {}).get("clean_power_source"),
        "clear_times_source": (extra or {}).get("clear_times_source"),
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


def _reconstruct_series_from_matrix(dh: object, matrix_attr: str, power_col: str) -> tuple[pd.Series, str] | None:
    matrix = getattr(dh, matrix_attr, None)
    df = getattr(dh, "data_frame", None)
    day_index = getattr(dh, "day_index", None)

    if not isinstance(matrix, np.ndarray) or not isinstance(df, pd.DataFrame) or not isinstance(df.index, pd.DatetimeIndex):
        return None
    if "seq_index" not in df.columns or not isinstance(day_index, pd.DatetimeIndex):
        return None

    day_lookup = {pd.Timestamp(d).normalize(): i for i, d in enumerate(day_index)}

    values = []
    for ts, seq in zip(df.index, df["seq_index"], strict=False):
        day_pos = day_lookup.get(pd.Timestamp(ts).normalize())
        if day_pos is None:
            values.append(np.nan)
            continue
        row = int(seq)
        if row < 0 or day_pos < 0 or row >= matrix.shape[0] or day_pos >= matrix.shape[1]:
            values.append(np.nan)
            continue
        values.append(matrix[row, day_pos])

    series = pd.Series(values, index=df.index, name="ac_power_clean")
    if series.index.tz is None:
        series.index = series.index.tz_localize(INTERNAL_TZ)
    else:
        series.index = series.index.tz_convert(INTERNAL_TZ)
    return series, f"dh.{matrix_attr}+dh.day_index+dh.data_frame.seq_index"


def extract_clean_power_series(dh: object, power_col: str = "ac_power") -> tuple[pd.Series, str]:
    """Extract SDT shift-corrected clean power series deterministically."""
    df = getattr(dh, "data_frame", None)
    if isinstance(df, pd.DataFrame) and isinstance(df.index, pd.DatetimeIndex):
        scored_cols: list[tuple[int, str]] = []
        for col in df.columns:
            lname = str(col).lower()
            score = 0
            if str(col) == power_col:
                score += 10
            if power_col.lower() in lname or "power" in lname:
                score += 3
            if any(k in lname for k in ("clean", "filled", "processed", "fixed", "shift")):
                score += 5
            if score > 0 and pd.api.types.is_numeric_dtype(df[col]):
                scored_cols.append((score, str(col)))

        if scored_cols:
            scored_cols.sort(key=lambda x: (-x[0], x[1]))
            selected_col = scored_cols[0][1]
            series = pd.to_numeric(df[selected_col], errors="coerce").rename("ac_power_clean")
            if series.index.tz is None:
                series.index = series.index.tz_localize(INTERNAL_TZ)
            else:
                series.index = series.index.tz_convert(INTERNAL_TZ)
            return series, f"dh.data_frame[{selected_col!r}]"

    for matrix_attr in ("filled_data_matrix", "processed_data_matrix"):
        reconstructed = _reconstruct_series_from_matrix(dh, matrix_attr=matrix_attr, power_col=power_col)
        if reconstructed is not None:
            return reconstructed

    raise RuntimeError(
        "Unable to identify SDT cleaned shift-corrected power series through public attributes. "
        "Expected one of: numeric post-pipeline power column in dh.data_frame OR "
        "reconstructable filled/processed matrix with day_index+seq_index."
    )


def _ensure_clear_times(dh: object) -> tuple[pd.Series, str]:
    """Compute and retrieve clear-times mask using SDT v2.1.x public API."""
    attempted: list[str] = []

    def _from_boolean_masks() -> tuple[pd.Series, str] | None:
        bm = getattr(dh, "boolean_masks", None)
        if bm is None:
            return None
        for attr in ("clear_times", "clear_time", "clear"):
            if hasattr(bm, attr):
                values = getattr(bm, attr)
                if values is not None:
                    if isinstance(values, pd.Series):
                        series = values.astype(bool)
                    else:
                        series = pd.Series(np.asarray(values).astype(bool), index=getattr(dh, "data_frame").index)
                    return series, f"dh.boolean_masks.{attr}"
        return None

    found = _from_boolean_masks()
    if found is not None:
        return found

    for method_name in (
        "make_filled_data_matrix",
        "find_clipped_times",
        "get_daily_flags",
        "calculate_scsf_performance_index",
    ):
        method = getattr(dh, method_name, None)
        if callable(method):
            attempted.append(method_name)
            try:
                method()
            except TypeError:
                try:
                    method(return_values=True)
                except Exception:
                    pass
            except Exception:
                pass
            found = _from_boolean_masks()
            if found is not None:
                return found

    bm = getattr(dh, "boolean_masks", None)
    bm_attrs = [a for a in dir(bm) if not a.startswith("_")] if bm is not None else []
    raise RuntimeError(
        "Could not compute clear-times mask via SDT public API. "
        f"Attempted methods={attempted}, boolean_masks_attrs={bm_attrs}"
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

    clear_times_source = None
    clean_source = "unresolved"
    if out_dir is not None:
        write_sdt_introspect(
            dh,
            out_dir=out_dir,
            power_col=power_col,
            extra={"stage": "after_run_pipeline", "clean_power_source": clean_source, "clear_times_source": clear_times_source},
        )

    clear_col, clear_times_source = _ensure_clear_times(dh)
    if hasattr(dh, "augment_data_frame") and callable(getattr(dh, "augment_data_frame")):
        dh.augment_data_frame(clear_col, "is_clear_time")

    clear_col = clear_col.reindex(parsed.index).fillna(False).astype(bool)
    clear_col.name = "is_clear_time"

    if out_dir is not None:
        _write_parquet(clear_col.to_frame(), Path(out_dir) / "03_clear_times_mask.parquet")

    report = None
    try:
        ac_power_clean, clean_source = extract_clean_power_series(dh, power_col=power_col)
        clean_error = None
    except Exception as exc:
        clean_error = str(exc)
        if out_dir is not None:
            write_sdt_introspect(
                dh,
                out_dir=out_dir,
                power_col=power_col,
                extra={
                    "stage": "extract_clean_power_series",
                    "error": str(exc),
                    "clean_power_source": clean_source,
                    "clear_times_source": clear_times_source,
                },
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
        "clear_times_source": clear_times_source,
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
