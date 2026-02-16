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
        "clear_days_source": (extra or {}).get("clear_days_source"),
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

    daily_flags = getattr(dh, "daily_flags", None)
    if daily_flags is not None:
        try:
            daily_attrs = vars(daily_flags)
        except Exception:
            daily_attrs = {}
        for key, value in daily_attrs.items():
            stats = _bool_stats(value)
            if stats is not None:
                out["boolean_masks_diagnostics"].append(
                    {
                        "name": f"daily_flags.{key}",
                        "type": type(value).__name__,
                        "shape": _shape_for(value),
                        "counts": stats,
                    }
                )

    df_masks = getattr(dh, "data_frame", None)
    if isinstance(df_masks, pd.DataFrame):
        for col in ("is_fit_time", "is_clipped_time"):
            if col in df_masks.columns:
                stats = _bool_stats(df_masks[col])
                if stats is not None:
                    out["boolean_masks_diagnostics"].append(
                        {
                            "name": f"data_frame.{col}",
                            "type": "Series",
                            "shape": _shape_for(df_masks[col]),
                            "counts": stats,
                        }
                    )

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


def _coerce_bool_day_flags(value: object, n_days: int) -> np.ndarray | None:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return None
    flat = arr.reshape(-1)
    if flat.size != n_days:
        return None
    if flat.dtype == np.bool_:
        return flat
    if np.issubdtype(flat.dtype, np.number):
        return flat.astype(float) > 0
    series = pd.Series(flat)
    mapped = series.map(
        {
            True: True,
            False: False,
            "true": True,
            "false": False,
            "clear": True,
            "not_clear": False,
            "1": True,
            "0": False,
        }
    )
    if mapped.isna().all():
        return None
    return mapped.fillna(False).to_numpy(dtype=bool)


def _safe_solver_name(value: object, default: str = "OSQP") -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def run_clear_day_detection(dh: object, solver: str = "OSQP") -> str:
    solver_name = _safe_solver_name(solver)
    detect_clear_days = getattr(dh, "detect_clear_days", None)
    if callable(detect_clear_days):
        try:
            detect_clear_days(solver=solver_name)
            return "detect_clear_days"
        except TypeError:
            detect_clear_days()
            return "detect_clear_days"
        except Exception as exc:
            LOGGER.warning("detect_clear_days failed: %s", exc)

    detect_clear_sky = getattr(dh, "detect_clear_sky", None)
    if callable(detect_clear_sky):
        try:
            detect_clear_sky(solver=solver_name)
            return "detect_clear_sky"
        except TypeError:
            detect_clear_sky()
            return "detect_clear_sky"
        except Exception as exc:
            LOGGER.warning("detect_clear_sky failed: %s", exc)

    LOGGER.warning("clear day detection not available in this SDT DataHandler")
    return "unavailable"


def get_clear_day_flags(dh: object) -> tuple[np.ndarray | None, str]:
    matrix = getattr(dh, "filled_data_matrix", None)
    if hasattr(matrix, "shape") and len(matrix.shape) == 2:
        n_days = int(matrix.shape[1])
    else:
        day_index = getattr(dh, "day_index", None)
        n_days = int(len(day_index)) if isinstance(day_index, pd.DatetimeIndex) else 0
    if n_days <= 0:
        return None, "unavailable"

    keys = ("clear", "clear_day", "clear_days", "clear_sky", "is_clear")

    get_daily_flags = getattr(dh, "get_daily_flags", None)
    if callable(get_daily_flags):
        try:
            flags = get_daily_flags()
            if isinstance(flags, dict):
                for key in keys:
                    if key in flags:
                        arr = _coerce_bool_day_flags(flags[key], n_days)
                        if arr is not None:
                            return arr, f"sdt:get_daily_flags:{key}"
            if isinstance(flags, pd.DataFrame):
                for key in keys:
                    if key in flags.columns:
                        arr = _coerce_bool_day_flags(flags[key].to_numpy(), n_days)
                        if arr is not None:
                            return arr, f"sdt:get_daily_flags:{key}"
            if isinstance(flags, pd.Series):
                arr = _coerce_bool_day_flags(flags.to_numpy(), n_days)
                if arr is not None:
                    return arr, "sdt:get_daily_flags:series"
        except Exception as exc:
            LOGGER.warning("get_daily_flags failed during clear-day extraction: %s", exc)

    daily_flags = getattr(dh, "daily_flags", None)
    if daily_flags is not None:
        try:
            attrs = vars(daily_flags)
        except Exception:
            attrs = {}
        for key in keys:
            if key in attrs:
                arr = _coerce_bool_day_flags(attrs[key], n_days)
                if arr is not None:
                    return arr, f"sdt:daily_flags:{key}"
        for key, val in attrs.items():
            arr = _coerce_bool_day_flags(val, n_days)
            if arr is not None:
                return arr, f"sdt:daily_flags:{key}"

    return None, "unavailable"


def _all_false_clipped_series(dh: object) -> pd.Series:
    df = getattr(dh, "data_frame", None)
    if isinstance(df, pd.DataFrame) and isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(False, index=df.index, name="is_clipped_time")
    return pd.Series(dtype=bool, name="is_clipped_time")


def run_clipping_detection_safe(dh: object, solver: str) -> tuple[np.ndarray | None, str | None]:
    solver_name = _safe_solver_name(solver)
    find_clipped_times = getattr(dh, "find_clipped_times", None)
    if not callable(find_clipped_times):
        return None, "find_clipped_times_not_available"
    try:
        find_clipped_times(solver=solver_name)
    except TypeError:
        try:
            find_clipped_times()
        except MemoryError as exc:
            LOGGER.warning("SDT clipping detection MemoryError: %s", exc)
            return None, str(exc)
        except Exception as exc:
            LOGGER.warning("SDT clipping detection failed: %s", exc)
            return None, str(exc)
    except MemoryError as exc:
        LOGGER.warning("SDT clipping detection MemoryError: %s", exc)
        return None, str(exc)
    except Exception as exc:
        LOGGER.warning("SDT clipping detection failed: %s", exc)
        return None, str(exc)

    bm = getattr(dh, "boolean_masks", None)
    clipped = getattr(bm, "clipped_times", None) if bm is not None else None
    if isinstance(clipped, np.ndarray):
        return clipped, None
    return None, "clipped_times_not_generated"


def _build_fit_times(dh: object, solver: str) -> tuple[pd.Series, str, int, float]:
    bm = getattr(dh, "boolean_masks", None)
    if bm is None:
        df = getattr(dh, "data_frame", None)
        if isinstance(df, pd.DataFrame):
            return pd.Series(False, index=df.index, name="is_fit_time"), "unavailable", 0, 0.0
        raise RuntimeError("Unable to build fit times without boolean_masks and data_frame")

    run_clear_day_detection(dh, solver=solver)
    clear_days, source = get_clear_day_flags(dh)

    daytime = np.asarray(getattr(bm, "daytime"), dtype=bool)
    n_rows, n_days = daytime.shape

    if clear_days is not None and clear_days.size == n_days:
        fit_times_2d = daytime & clear_days[None, :]
        clear_days_source = source.replace("sdt:get_daily_flags", "sdt:daily_flags")
    else:
        fit_times_2d = daytime
        clear_days = np.zeros(n_days, dtype=bool)
        clear_days_source = "fallback:daytime_only"
        LOGGER.warning("No clear-day flags available, using daytime-only fit mask")

    fit_series = _augment_mask_to_data_frame(dh, fit_times_2d, "is_fit_time")
    fit_series = fit_series.astype(bool)
    n_clear_days = int(np.count_nonzero(clear_days))
    clear_day_fraction = float(n_clear_days / max(n_days, 1))
    return fit_series, clear_days_source, n_clear_days, clear_day_fraction

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
    config: dict[str, Any] | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run Block A (A1/A2/A3) and return artifacts and summaries."""
    from solardatatools import DataHandler

    if not isinstance(ac_power.index, pd.DatetimeIndex):
        raise ValueError("ac_power index must be DatetimeIndex.")

    ac_power = ac_power.copy()
    if str(ac_power.index.tz) != INTERNAL_TZ:
        ac_power.index = ac_power.index.tz_convert(INTERNAL_TZ)

    parsed = ac_power.rename(power_col).to_frame()

    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        _write_parquet(parsed, out_path / "01_parsed_tzaware.parquet")

    dh = DataHandler(parsed)
    pipeline_cfg = (config or {}).get("pipeline", {})
    solver_name = _safe_solver_name(pipeline_cfg.get("solver", "CLARABEL"), default="CLARABEL")
    fix_shifts = bool(pipeline_cfg.get("fix_shifts", True))
    verbose_sdt = bool(pipeline_cfg.get("verbose_sdt", False))
    dh.run_pipeline(power_col=power_col, fix_shifts=fix_shifts, solver=solver_name, verbose=verbose_sdt)

    skip_clipping = bool(pipeline_cfg.get("skip_clipping", True))

    clear_days_source = "unavailable"
    clean_source = "unresolved"
    if _report_no_corrections(dh) and isinstance(getattr(dh, "data_frame", None), pd.DataFrame):
        if power_col in dh.data_frame.columns:
            clean_source = f"data_frame:{power_col}:no_corrections"

    if out_dir is not None:
        write_sdt_introspect(
            dh,
            out_dir=out_dir,
            power_col=power_col,
            extra={"stage": "after_run_pipeline", "clean_power_source": clean_source, "clear_days_source": clear_days_source},
        )

    fit_col, clear_days_source, n_clear_days, clear_day_fraction = _build_fit_times(dh, solver=solver_name)
    fit_col = fit_col.reindex(parsed.index).fillna(False).astype(bool)
    fit_col.name = "is_fit_time"
    LOGGER.info("fit_times created via %s with %d true points", clear_days_source, int(fit_col.sum()))

    if out_dir is not None:
        _write_parquet(fit_col.to_frame(), Path(out_dir) / "03_fit_times_mask.parquet")

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
                    "clear_days_source": clear_days_source,
                },
            )
        raise
    finally:
        try:
            report = dh.report(return_values=True)
        except Exception as exc:  # pragma: no cover
            report = {"error": str(exc)}

    fit_col = fit_col.reindex(ac_power_clean.index).fillna(False).astype(bool)
    fit_col.name = "is_fit_time"

    clipping_detection_used = False
    clipping_detection_failed = False
    clipping_detection_error = None

    clipped: pd.Series
    if skip_clipping:
        LOGGER.info("Skipping SDT clipping detection by config")
        clipped = _all_false_clipped_series(dh).reindex(ac_power_clean.index).fillna(False).astype(bool)
    else:
        clipping_detection_used = True
        clipped_2d, clipping_error = run_clipping_detection_safe(dh, solver=solver_name)
        if clipped_2d is None:
            clipping_detection_failed = True
            clipping_detection_error = clipping_error
            LOGGER.warning("Clipping detection failed; using no-clipping mask.")
            clipped = _all_false_clipped_series(dh).reindex(ac_power_clean.index).fillna(False).astype(bool)
        else:
            clipped = _augment_mask_to_data_frame(dh, clipped_2d, "is_clipped_time")
            clipped = clipped.reindex(ac_power_clean.index).fillna(False).astype(bool)

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
    n_fit_times = int(fit_col.sum())

    sdt_summary = {
        "n_days_total": n_days_total,
        "n_clear_days": n_clear_days,
        "clear_day_fraction": clear_day_fraction,
        "n_fit_times": n_fit_times,
        "clear_days_source": clear_days_source,
        "skip_clipping": skip_clipping,
        "clipping_detection_used": clipping_detection_used,
        "clipping_detection_failed": clipping_detection_failed,
        "clipping_detection_error": clipping_detection_error,
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
        "clear_days_source": clear_days_source,
        "fit_times_counts": {"true": int(fit_col.sum()), "false": int((~fit_col).sum())},
        "clipped_times_counts": {"true": int(clipped.sum()), "false": int((~clipped).sum())},
    }

    raw_matrix = getattr(dh, "raw_data_matrix", None)
    filled_matrix = getattr(dh, "filled_data_matrix", None)
    day_index = getattr(dh, "day_index", None)
    seq_index = getattr(dh, "data_frame", pd.DataFrame()).get("seq_index") if isinstance(getattr(dh, "data_frame", None), pd.DataFrame) else None
    sdt_daily_flags = None
    if hasattr(dh, "get_daily_flags") and callable(getattr(dh, "get_daily_flags")):
        try:
            flags = dh.get_daily_flags()
            if isinstance(flags, dict):
                sdt_daily_flags = pd.DataFrame(flags)
            elif isinstance(flags, pd.DataFrame):
                sdt_daily_flags = flags.copy()
        except Exception:
            sdt_daily_flags = None

    filled_timeseries = None
    if isinstance(filled_matrix, np.ndarray) and isinstance(day_index, pd.DatetimeIndex) and isinstance(seq_index, pd.Series):
        reconstructed = _reconstruct_series_from_matrix(dh, matrix_attr="filled_data_matrix")
        if reconstructed is not None:
            filled_timeseries = reconstructed[0].to_frame(name="filled_power")

    return {
        "parsed": parsed,
        "ac_power_clean": ac_power_clean.to_frame(),
        "fit_times": fit_col.to_frame(),
        "clipped_times": clipped.to_frame(name="is_clipped_time"),
        "daily_flags": daily_flags,
        "clipping_summary": clipping_summary,
        "sdt_summary": sdt_summary,
        "sdt_introspect": introspect,
        "raw_data_matrix": pd.DataFrame(raw_matrix) if isinstance(raw_matrix, np.ndarray) else pd.DataFrame(),
        "filled_data_matrix": pd.DataFrame(filled_matrix) if isinstance(filled_matrix, np.ndarray) else pd.DataFrame(),
        "sdt_daily_flags": sdt_daily_flags if isinstance(sdt_daily_flags, pd.DataFrame) else pd.DataFrame(),
        "filled_timeseries": filled_timeseries if isinstance(filled_timeseries, pd.DataFrame) else pd.DataFrame(),
    }


def apply_exclusion_rules(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    """Apply A3 exclusion logic."""
    pconf = config.get("pipeline", {})
    clip_share_thr = float(pconf.get("clipping_threshold_day_share", 0.10))
    clip_median_thr = float(pconf.get("clipping_fraction_day_median", 0.01))
    if "clear_time_fraction_min" in pconf:
        LOGGER.warning("pipeline.clear_time_fraction_min is deprecated and ignored.")

    clipping_day_share_mean = float(summary.get("clipping_day_share_mean", 0.0))
    clipping_fraction_day_median = float(summary.get("clipping_fraction_day_median", 0.0))
    n_fit_times = int(summary.get("n_fit_times", 0))

    time_shift_applied = summary.get("time_shift_correction_applied")
    tz_corr = summary.get("time_zone_correction")

    suspect_large_shift = False
    if bool(time_shift_applied) and isinstance(tz_corr, (int, float)):
        suspect_large_shift = abs(float(tz_corr)) >= 2.0

    exclude_low_clear = n_fit_times <= 0

    return {
        "exclude_clipping": clipping_day_share_mean >= clip_share_thr or clipping_fraction_day_median >= clip_median_thr,
        "exclude_low_clear": exclude_low_clear,
        "suspect_large_shift": suspect_large_shift,
    }
