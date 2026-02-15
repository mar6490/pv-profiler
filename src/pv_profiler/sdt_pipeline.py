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


def _bool_stats(value: object) -> dict[str, int] | None:
    try:
        if isinstance(value, pd.Series):
            arr = value.to_numpy()
        elif isinstance(value, np.ndarray):
            arr = value
        elif isinstance(value, (list, tuple)):
            arr = np.asarray(value)
        else:
            return None
        flat = np.ravel(arr)
        if flat.size == 0 or flat.size > 1_000_000:
            return None
        series = pd.Series(flat)
        n_true = int((series == True).sum())  # noqa: E712
        n_false = int((series == False).sum())  # noqa: E712
        n_nan = int(series.isna().sum())
        return {"true": n_true, "false": n_false, "nan": n_nan}
    except Exception:  # pragma: no cover
        return None


def _report_no_corrections(dh: object) -> bool:
    try:
        report = dh.report(return_values=True)
    except Exception:
        return False
    if not isinstance(report, dict):
        return False
    ts_ok = report.get("time shift correction") is False
    tz_val = report.get("time zone correction")
    try:
        tz_ok = float(tz_val) == 0.0
    except Exception:
        tz_ok = False
    return ts_ok and tz_ok


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
        "boolean_masks_diagnostics": [],
        "report": None,
        "extra": extra or {},
        "clean_power_source": (extra or {}).get("clean_power_source"),
        "clear_times_source": (extra or {}).get("clear_times_source"),
        "detect_clear_sky_available": bool(callable(getattr(dh, "detect_clear_sky", None))),
        "detect_clear_days_available": bool(callable(getattr(dh, "detect_clear_days", None))),
        "boolean_masks_dir": [],
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

    bm = getattr(dh, "boolean_masks", None)
    if bm is not None:
        out["boolean_masks_dir"] = [name for name in dir(bm) if not name.startswith("_")]
        for name in dir(bm):
            if name.startswith("_"):
                continue
            try:
                value = getattr(bm, name)
            except Exception as exc:  # pragma: no cover
                out["boolean_masks_diagnostics"].append({"name": name, "type": "<error>", "shape": str(exc)})
                continue
            item = {"name": name, "type": type(value).__name__, "shape": _shape_for(value)}
            if isinstance(value, np.ndarray):
                item["dtype"] = str(value.dtype)
                if value.size:
                    item["counts"] = {
                        "true": int(np.count_nonzero(value == True)),  # noqa: E712
                        "false": int(np.count_nonzero(value == False)),  # noqa: E712
                    }
            out["boolean_masks_diagnostics"].append(item)

    try:
        out["report"] = dh.report(return_values=True)
    except Exception as exc:  # pragma: no cover
        out["report"] = {"error": str(exc)}

    p = Path(out_dir) / "07_sdt_introspect.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return out


def _mask_to_series(mask: object, index: pd.DatetimeIndex) -> tuple[pd.Series, str]:
    if isinstance(mask, pd.Series):
        series = mask.reindex(index)
    else:
        arr = np.asarray(mask)
        flat = np.ravel(arr)
        if flat.size != len(index):
            raise ValueError(f"mask size {flat.size} does not match index length {len(index)}")
        series = pd.Series(flat, index=index)
    unique_sample = pd.Series(series).dropna().astype(str).unique()[:10].tolist()
    return series.fillna(False).astype(bool), f"sample_unique={unique_sample}"


def _augment_mask_to_data_frame(dh: object, mask: object, target_col: str) -> pd.Series:
    df = getattr(dh, "data_frame", None)
    if not isinstance(df, pd.DataFrame) or not isinstance(df.index, pd.DatetimeIndex):
        raise RuntimeError("dh.data_frame with DatetimeIndex is required for mask augmentation.")
    sample_info = f"mask_shape={_shape_for(mask)}"
    if hasattr(dh, "augment_data_frame") and callable(getattr(dh, "augment_data_frame")):
        try:
            dh.augment_data_frame(mask, target_col)
        except Exception as exc:
            raise RuntimeError(
                f"augment_data_frame failed for {target_col}; mask_type={type(mask).__name__}, "
                f"{sample_info}, error={exc}"
            ) from exc

    df2 = getattr(dh, "data_frame", None)
    if isinstance(df2, pd.DataFrame) and target_col in df2.columns:
        return df2[target_col].reindex(df.index).fillna(False).astype(bool)
    series, _ = _mask_to_series(mask, df.index)
    return series


def _reconstruct_series_from_matrix(dh: object, matrix_attr: str) -> tuple[pd.Series, str] | None:
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
        # Required fast path: no corrections -> use power column directly.
        if _report_no_corrections(dh) and power_col in df.columns:
            series = pd.to_numeric(df[power_col], errors="coerce").rename("ac_power_clean")
            if series.index.tz is None:
                series.index = series.index.tz_localize(INTERNAL_TZ)
            else:
                series.index = series.index.tz_convert(INTERNAL_TZ)
            return series, f"data_frame:{power_col}:no_corrections"

        scored_cols: list[tuple[int, str]] = []
        for col in df.columns:
            lname = str(col).lower()
            if not any(k in lname for k in ("clean", "filled", "processed", "fixed", "shift")):
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            score = 5 + (3 if (power_col.lower() in lname or "power" in lname) else 0)
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
        reconstructed = _reconstruct_series_from_matrix(dh, matrix_attr=matrix_attr)
        if reconstructed is not None:
            return reconstructed

    raise RuntimeError(
        "Unable to identify SDT cleaned shift-corrected power series through public attributes. "
        "Expected one of: no-corrections data_frame power_col, explicit post-pipeline column, "
        "or reconstructable filled/processed matrix with day_index+seq_index."
    )


def _ensure_optional_mask(dh: object, attr_name: str, target_col: str) -> tuple[pd.Series, str] | None:
    bm = getattr(dh, "boolean_masks", None)
    if bm is None or not hasattr(bm, attr_name):
        return None
    values = getattr(bm, attr_name)
    if values is None:
        return None
    series = _augment_mask_to_data_frame(dh, values, target_col)
    return series, f"dh.boolean_masks.{attr_name}"


def _ensure_clear_times(dh: object) -> tuple[pd.Series, str]:
    """Compute clear-times mask with robust staged fallbacks (never hard-fail)."""
    bm = getattr(dh, "boolean_masks", None)
    matrix = getattr(dh, "filled_data_matrix", None)
    matrix_shape = getattr(matrix, "shape", None)

    def _find_2d_candidates() -> list[tuple[str, np.ndarray, float]]:
        if bm is None or matrix_shape is None:
            return []
        candidates: list[tuple[str, np.ndarray, float]] = []
        for name in dir(bm):
            if name.startswith("_") or name in {"daytime", "missing_values", "infill"}:
                continue
            try:
                value = getattr(bm, name)
            except Exception:
                continue
            if isinstance(value, np.ndarray) and value.dtype == np.bool_ and value.shape == matrix_shape:
                density = float(np.mean(value)) if value.size else 0.0
                candidates.append((name, value, density))
        return candidates

    # Stage A: detect_clear_sky + inferred newly-generated clear-time 2D mask.
    detect_clear_sky = getattr(dh, "detect_clear_sky", None)
    if callable(detect_clear_sky):
        try:
            detect_clear_sky()
        except Exception as exc:
            LOGGER.warning("detect_clear_sky failed; continuing with fallback strategy: %s", exc)

        candidates = _find_2d_candidates()
        if candidates:
            selected_name = None
            selected_mask = None
            if len(candidates) == 1:
                selected_name, selected_mask, _ = candidates[0]
            else:
                plausible = [c for c in candidates if 0.005 <= c[2] <= 0.30]
                if plausible:
                    selected_name, selected_mask, _ = sorted(plausible, key=lambda c: (abs(c[2] - 0.10), c[0]))[0]
                else:
                    selected_name, selected_mask, _ = sorted(candidates, key=lambda c: (c[2], c[0]))[0]
            if selected_mask is not None:
                series = _augment_mask_to_data_frame(dh, selected_mask, "is_clear_time")
                return series.astype(bool), f"detect_clear_sky:boolean_masks.{selected_name}"

    # Stage B: detect_clear_days + tile to 2D and intersect with daytime/non-missing.
    detect_clear_days = getattr(dh, "detect_clear_days", None)
    if callable(detect_clear_days) and bm is not None and matrix_shape is not None:
        try:
            clear_days = detect_clear_days()
            clear_days_1d = np.asarray(clear_days).reshape(-1).astype(bool)
            n_rows, n_days = matrix_shape
            if clear_days_1d.size == n_days:
                daytime = np.asarray(getattr(bm, "daytime"), dtype=bool)
                missing = np.asarray(getattr(bm, "missing_values"), dtype=bool)
                clear_times_2d = np.tile(clear_days_1d, (n_rows, 1)) & daytime & ~missing
                series = _augment_mask_to_data_frame(dh, clear_times_2d, "is_clear_time")
                return series.astype(bool), "detect_clear_days:tiled&daytime&~missing"
            LOGGER.warning(
                "detect_clear_days returned %d days, expected %d; using daytime fallback",
                clear_days_1d.size,
                n_days,
            )
        except Exception as exc:
            LOGGER.warning("detect_clear_days failed; using daytime fallback: %s", exc)

    # Stage C: deterministic daytime-only fallback.
    if bm is not None:
        try:
            daytime = np.asarray(getattr(bm, "daytime"), dtype=bool)
            missing = np.asarray(getattr(bm, "missing_values"), dtype=bool)
            daytime_mask = daytime & ~missing
            series = _augment_mask_to_data_frame(dh, daytime_mask, "is_clear_time")
            return series.astype(bool), "fallback:daytime_only"
        except Exception as exc:
            LOGGER.warning("daytime fallback failed to augment; creating series directly: %s", exc)

    df = getattr(dh, "data_frame", None)
    if isinstance(df, pd.DataFrame):
        return pd.Series(False, index=df.index, name="is_clear_time"), "fallback:all_false"
    raise RuntimeError("Unable to create is_clear_time mask: missing dh.data_frame")


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
    if _report_no_corrections(dh) and isinstance(getattr(dh, "data_frame", None), pd.DataFrame):
        if power_col in dh.data_frame.columns:
            clean_source = f"data_frame:{power_col}:no_corrections"

    if out_dir is not None:
        write_sdt_introspect(
            dh,
            out_dir=out_dir,
            power_col=power_col,
            extra={"stage": "after_run_pipeline", "clean_power_source": clean_source, "clear_times_source": clear_times_source},
        )

    clear_col, clear_times_source = _ensure_clear_times(dh)
    clear_col = clear_col.reindex(parsed.index).fillna(False).astype(bool)
    clear_col.name = "is_clear_time"
    LOGGER.info("clear_times created via %s with %d true points", clear_times_source, int(clear_col.sum()))

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

    clipped_info = _ensure_optional_mask(dh, "clipped_times", "is_clipped_time")
    if clipped_info is not None:
        clipped, _clipped_source = clipped_info
        clipped = clipped.reindex(ac_power_clean.index).fillna(False).astype(bool)
    else:
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
        "clear_times_source": clear_times_source,
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
        "clipped_times": clipped.to_frame(name="is_clipped_time"),
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

    exclude_low_clear = clear_time_fraction < clear_frac_min
    if str(summary.get("clear_times_source", "")).startswith("fallback:daytime_only"):
        exclude_low_clear = True

    return {
        "exclude_clipping": clipping_day_share_mean >= clip_share_thr or clipping_fraction_day_median >= clip_median_thr,
        "exclude_low_clear": exclude_low_clear,
        "suspect_large_shift": suspect_large_shift,
    }
