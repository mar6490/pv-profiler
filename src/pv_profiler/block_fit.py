from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .types import Block3Result


BOOL_COLS = ["clear", "cloudy", "density", "inverter_clipped", "linearity", "no_errors"]
RULE_USED = "(clear == True) AND (inverter_clipped == False) AND (no_errors == True)"


def _to_bool_series(series: pd.Series) -> pd.Series:
    mapping = {
        True: True,
        False: False,
        "True": True,
        "False": False,
        "true": True,
        "false": False,
        1: True,
        0: False,
        "1": True,
        "0": False,
    }
    out = series.map(lambda x: mapping.get(x, x))
    if out.dtype != bool:
        out = out.fillna(False).astype(bool)
    return out


def load_daily_flags(flags_csv: str | Path) -> pd.DataFrame:
    df_flags = pd.read_csv(flags_csv, index_col=0)
    df_flags.index = pd.to_datetime(df_flags.index, errors="coerce").normalize()
    df_flags = df_flags[~df_flags.index.isna()].sort_index()

    for col in BOOL_COLS:
        if col in df_flags.columns:
            df_flags[col] = _to_bool_series(df_flags[col])

    if "capacity_cluster" in df_flags.columns:
        df_flags["capacity_cluster"] = pd.to_numeric(df_flags["capacity_cluster"], errors="coerce")

    required = ["clear", "inverter_clipped", "no_errors"]
    missing = [c for c in required if c not in df_flags.columns]
    if missing:
        raise ValueError(f"Missing required daily flags columns: {missing}")

    df_flags["is_fit_day"] = (
        (df_flags["clear"] == True)
        & (df_flags["inverter_clipped"] == False)
        & (df_flags["no_errors"] == True)
    )
    return df_flags


def run_block3_fit_selection(
    power_df: pd.DataFrame,
    daily_flags_df: pd.DataFrame,
    *,
    fit_mode: str = "mask_to_nan",
    min_fit_days: int = 10,
) -> Block3Result:
    if not isinstance(power_df.index, pd.DatetimeIndex):
        raise ValueError("power_df index must be a DatetimeIndex")
    if "power" not in power_df.columns:
        raise ValueError("power_df must contain column 'power'")

    if "is_fit_day" not in daily_flags_df.columns:
        raise ValueError("daily_flags_df must contain column 'is_fit_day'")

    power = power_df[["power"]].copy().sort_index()
    flags = daily_flags_df.copy().sort_index()

    power_days = pd.DatetimeIndex(power.index.normalize().unique())
    unmatched_days = int((~power_days.isin(flags.index)).sum())

    fit_day_series = flags["is_fit_day"].astype(bool)
    fit_mask = power.index.normalize().map(fit_day_series).fillna(False).astype(bool)

    n_fit_days = int(fit_day_series.sum())
    n_days_total = int(len(flags))

    if n_fit_days < min_fit_days:
        return Block3Result(
            status="insufficient_fit_days",
            n_fit_days=n_fit_days,
            min_required_fit_days=min_fit_days,
            n_days_total=n_days_total,
            n_unmatched_days=unmatched_days,
            rule_used=RULE_USED,
            fit_mode=fit_mode,
            fit_days_df=flags,
            power_fit_df=None,
        )

    if fit_mode == "mask_to_nan":
        df_fit = power.copy()
        df_fit.loc[~fit_mask, "power"] = np.nan
    elif fit_mode == "filter_rows":
        df_fit = power.loc[fit_mask].copy()
    else:
        raise ValueError("fit_mode must be one of: mask_to_nan, filter_rows")

    return Block3Result(
        status="ok",
        n_fit_days=n_fit_days,
        min_required_fit_days=min_fit_days,
        n_days_total=n_days_total,
        n_unmatched_days=unmatched_days,
        rule_used=RULE_USED,
        fit_mode=fit_mode,
        fit_days_df=flags,
        power_fit_df=df_fit,
    )


def write_block3_artifacts(result: Block3Result, output_dir: str | Path) -> dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    if result.fit_days_df is not None:
        p = out_dir / "03_fit_days.csv"
        result.fit_days_df.to_csv(p, index=True)
        written["fit_days"] = str(p)

    status_payload = {
        "status": result.status,
        "n_fit_days": result.n_fit_days,
        "min_required_fit_days": result.min_required_fit_days,
        "rule_used": result.rule_used,
        "n_days_total": result.n_days_total,
        "n_unmatched_days": result.n_unmatched_days,
        "fit_mode": result.fit_mode,
    }
    p_status = out_dir / "03_fit_status.json"
    p_status.write_text(json.dumps(status_payload, indent=2) + "\n", encoding="utf-8")
    written["fit_status"] = str(p_status)

    if result.status == "ok" and result.power_fit_df is not None:
        summary = {
            "n_days_total": result.n_days_total,
            "n_fit_days": result.n_fit_days,
            "share_fit_days": (result.n_fit_days / result.n_days_total) if result.n_days_total else 0.0,
            "n_samples_total": int(result.n_samples_total),
            "n_samples_fit": int(result.n_samples_fit),
            "n_unmatched_days": result.n_unmatched_days,
            "fit_mode": result.fit_mode,
            "rule_used": result.rule_used,
        }
        p_summary = out_dir / "04_fit_mask_summary.json"
        p_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        written["fit_mask_summary"] = str(p_summary)

        p_fit = out_dir / "05_power_fit.parquet"
        result.power_fit_df.to_parquet(p_fit, index=True)
        written["power_fit"] = str(p_fit)

    return written
