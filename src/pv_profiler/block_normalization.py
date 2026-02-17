from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .types import NormalizationResult


def run_block4_normalize_from_parquet(
    input_power_fit_parquet: str | Path,
    output_dir: str | Path,
    quantile: float = 0.995,
    min_fit_samples_day: int = 1,
    dropna_output: bool = True,
) -> NormalizationResult:
    df = pd.read_parquet(input_power_fit_parquet)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input parquet must have DatetimeIndex.")
    if "power" not in df.columns:
        raise ValueError("Input parquet must contain column 'power'.")

    df = df[["power"]].copy().sort_index()
    day_key = df.index.normalize()

    grouped = df.groupby(day_key)["power"]
    p_peak_day = grouped.quantile(quantile)
    n_fit_samples_day = grouped.apply(lambda s: int(s.notna().sum()))

    daily_peak = pd.DataFrame(
        {
            "p_peak_day": p_peak_day,
            "n_fit_samples_day": n_fit_samples_day,
        }
    )
    daily_peak["is_usable_day"] = (
        (daily_peak["n_fit_samples_day"] >= int(min_fit_samples_day))
        & daily_peak["p_peak_day"].notna()
        & (daily_peak["p_peak_day"] > 0)
    )

    peak_map = daily_peak["p_peak_day"].where(daily_peak["is_usable_day"])
    df["p_norm"] = df["power"] / day_key.map(peak_map)

    if dropna_output:
        p_norm_out = df[["p_norm"]].dropna(subset=["p_norm"])
    else:
        p_norm_out = df[["p_norm"]].copy()

    usable = daily_peak[daily_peak["is_usable_day"]]
    gt1 = float((p_norm_out["p_norm"] > 1.0).mean()) if len(p_norm_out) else 0.0
    gt12 = float((p_norm_out["p_norm"] > 1.2).mean()) if len(p_norm_out) else 0.0

    diagnostics = {
        "n_days_total": int(day_key.nunique()),
        "n_fit_days": int((n_fit_samples_day > 0).sum()),
        "n_usable_days": int(daily_peak["is_usable_day"].sum()),
        "quantile_used": float(quantile),
        "n_samples_fit_total": int(df["power"].notna().sum()),
        "n_samples_norm_total": int(p_norm_out["p_norm"].notna().sum()),
        "share_p_norm_gt_1": gt1,
        "share_p_norm_gt_1_2": gt12,
        "min_p_peak_day": float(usable["p_peak_day"].min()) if not usable.empty else None,
        "max_p_peak_day": float(usable["p_peak_day"].max()) if not usable.empty else None,
        "decisions": [
            "peak computed from fit samples only (power non-null)",
            "days with peak<=0 excluded",
            f"dropna_output={dropna_output}",
        ],
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    daily_peak_path = out_dir / "06_daily_peak.csv"
    pnorm_path = out_dir / "07_p_norm_clear.parquet"
    diagnostics_path = out_dir / "07_p_norm_diagnostics.json"

    daily_peak_out = daily_peak.reset_index().rename(columns={"index": "date"})
    daily_peak_out["date"] = pd.to_datetime(daily_peak_out["date"]).dt.strftime("%Y-%m-%d")
    daily_peak_out.to_csv(daily_peak_path, index=False)
    p_norm_out.to_parquet(pnorm_path, index=True)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")

    return NormalizationResult(
        daily_peak=daily_peak,
        p_norm=p_norm_out,
        diagnostics=diagnostics,
        quantile=quantile,
        min_fit_samples_day=min_fit_samples_day,
        dropna_output=dropna_output,
    )
