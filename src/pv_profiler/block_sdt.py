from __future__ import annotations

import json
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .types import SdtBlockResult


def _ensure_power_frame(power_df: pd.DataFrame, power_col: str) -> pd.DataFrame:
    if not isinstance(power_df.index, pd.DatetimeIndex):
        raise ValueError("power_df index must be a DatetimeIndex for SDT onboarding.")
    if power_col not in power_df.columns:
        raise ValueError(f"power_df must contain power column '{power_col}'.")
    out = power_df[[power_col]].copy()
    if power_col != "power":
        out = out.rename(columns={power_col: "power"})
    return out.sort_index()


def _matrix_to_frame(matrix: Any) -> pd.DataFrame:
    if isinstance(matrix, pd.DataFrame):
        return matrix.copy()
    if isinstance(matrix, pd.Series):
        return matrix.to_frame()
    arr = np.asarray(matrix)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix-like structure, got shape {arr.shape}")
    return pd.DataFrame(arr)


def _daily_flags_to_frame(dh: Any) -> pd.DataFrame:
    flags = getattr(dh, "daily_flags", None)
    if flags is None:
        return pd.DataFrame()

    if isinstance(flags, pd.DataFrame):
        return flags.copy()

    if isinstance(flags, dict):
        source = flags
    else:
        source = getattr(flags, "__dict__", {})

    if not source:
        return pd.DataFrame()

    try:
        frame = pd.DataFrame(source)
    except ValueError:
        normalized = {k: pd.Series(v) for k, v in source.items()}
        frame = pd.DataFrame(normalized)
    if frame.empty:
        return frame

    frame = frame.reindex(sorted(frame.columns), axis=1)
    day_index = getattr(dh, "day_index", None)
    if isinstance(day_index, pd.Index) and len(day_index) == len(frame):
        frame.index = day_index
    return frame


def run_block2_sdt(
    power_df: pd.DataFrame,
    *,
    solver: str = "CLARABEL",
    fix_shifts: bool = True,
    power_col: str = "power",
) -> SdtBlockResult:
    data = _ensure_power_frame(power_df, power_col=power_col)

    from solardatatools import DataHandler  # type: ignore

    dh = DataHandler(data)

    report: dict[str, Any] | None = None
    report_error: str | None = None
    raw_matrix: pd.DataFrame | None = None
    filled_matrix: pd.DataFrame | None = None
    daily_flags: pd.DataFrame | None = None
    error_payload: dict[str, Any] | None = None
    status = "success"

    try:
        dh.run_pipeline(power_col="power", fix_shifts=fix_shifts, solver=solver)
    except MemoryError as exc:
        status = "failed"
        error_payload = {
            "exception_type": exc.__class__.__name__,
            "message": str(exc),
            "stack_summary": traceback.format_exc(limit=20),
            "stage": "run_pipeline",
        }
    except Exception as exc:  # pragma: no cover - runtime dependent
        status = "failed"
        error_payload = {
            "exception_type": exc.__class__.__name__,
            "message": str(exc),
            "stack_summary": traceback.format_exc(limit=20),
            "stage": "run_pipeline",
        }

    try:
        report = dh.report(return_values=True, verbose=False)
    except Exception as exc:
        report_error = f"Could not collect report: {exc.__class__.__name__}: {exc}"
        if status == "success":
            status = "partial"

    try:
        raw_attr = getattr(dh, "raw_data_matrix", None)
        if raw_attr is not None:
            raw_matrix = _matrix_to_frame(raw_attr)
    except Exception as exc:
        report_error = (report_error + " | " if report_error else "") + f"raw_data_matrix export failed: {exc}"
        if status == "success":
            status = "partial"

    try:
        filled_attr = getattr(dh, "filled_data_matrix", None)
        if filled_attr is not None:
            filled_matrix = _matrix_to_frame(filled_attr)
    except Exception as exc:
        report_error = (report_error + " | " if report_error else "") + f"filled_data_matrix export failed: {exc}"
        if status == "success":
            status = "partial"

    try:
        daily_flags = _daily_flags_to_frame(dh)
    except Exception as exc:
        report_error = (report_error + " | " if report_error else "") + f"daily_flags export failed: {exc}"
        if status == "success":
            status = "partial"

    if error_payload and any(x is not None for x in (report, raw_matrix, filled_matrix, daily_flags)):
        status = "partial"

    return SdtBlockResult(
        status=status,
        solver=solver,
        fix_shifts=fix_shifts,
        report=report,
        raw_data_matrix=raw_matrix,
        filled_data_matrix=filled_matrix,
        daily_flags=daily_flags,
        error=error_payload,
        message=report_error,
    )


def write_block2_artifacts(result: SdtBlockResult, output_dir: str | Path) -> dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, str] = {}

    if result.report is not None:
        p = out_dir / "02_sdt_report.json"
        p.write_text(json.dumps(result.report, indent=2, default=str) + "\n", encoding="utf-8")
        written["report"] = str(p)

    if result.daily_flags is not None and not result.daily_flags.empty:
        p = out_dir / "02_sdt_daily_flags.csv"
        result.daily_flags.to_csv(p, index=True)
        written["daily_flags"] = str(p)

    if result.raw_data_matrix is not None:
        p = out_dir / "02_sdt_raw_data_matrix.parquet"
        result.raw_data_matrix.to_parquet(p, index=True)
        written["raw_data_matrix"] = str(p)

    if result.filled_data_matrix is not None:
        p = out_dir / "02_sdt_filled_data_matrix.parquet"
        result.filled_data_matrix.to_parquet(p, index=True)
        written["filled_data_matrix"] = str(p)

    if result.error is not None:
        p = out_dir / "02_sdt_error.json"
        p.write_text(json.dumps(result.error, indent=2, default=str) + "\n", encoding="utf-8")
        written["error"] = str(p)

    status_path = out_dir / "02_sdt_status.json"
    status_payload = asdict(result)
    status_payload.pop("raw_data_matrix", None)
    status_payload.pop("filled_data_matrix", None)
    status_payload.pop("daily_flags", None)
    status_path.write_text(json.dumps(status_payload, indent=2, default=str) + "\n", encoding="utf-8")
    written["status"] = str(status_path)

    return written


def run_sdt_onboarding(power_series: pd.Series) -> tuple[pd.Series, list[str]]:
    """Backward-compatible helper used by existing run-single flow."""
    warnings: list[str] = []
    cleaned = power_series.copy().sort_index()

    inferred = pd.infer_freq(cleaned.index)
    if inferred is None:
        inferred = "5min"
        warnings.append("Could not infer frequency; using 5min default for regularization.")

    cleaned = cleaned.resample(inferred).mean().interpolate(limit=2)

    try:
        sdt_result = run_block2_sdt(cleaned.to_frame(name="power"))
        if sdt_result.status in {"failed", "partial"}:
            warnings.append(f"SDT onboarding status: {sdt_result.status}")
        if sdt_result.filled_data_matrix is not None and not sdt_result.filled_data_matrix.empty:
            values = sdt_result.filled_data_matrix.to_numpy().ravel(order="F")
            values = values[: len(cleaned)]
            cleaned = pd.Series(values, index=cleaned.index[: len(values)], name="power")
    except Exception as exc:  # pragma: no cover - defensive fallback
        warnings.append(f"SDT onboarding fallback used ({exc.__class__.__name__}: {exc}).")

    return cleaned, warnings
