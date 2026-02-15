"""Capacity estimation for effective kWp."""

from __future__ import annotations

from typing import Any

import pandas as pd


def estimate_kWp_effective(
    ac_power_clean: pd.Series,
    poa_cs: pd.Series,
    fit_mask: pd.Series,
    *,
    poa_threshold_wm2: float = 600.0,
) -> dict[str, Any]:
    """Estimate effective capacity from clean AC and best-model POA clearsky."""
    sel = (
        fit_mask.reindex(ac_power_clean.index).fillna(False)
        & ac_power_clean.notna()
        & poa_cs.reindex(ac_power_clean.index).notna()
        & (poa_cs.reindex(ac_power_clean.index) > poa_threshold_wm2)
    )

    ratios = ac_power_clean[sel] / (poa_cs[sel] / 1000.0)
    ratios = ratios.replace([float("inf"), float("-inf")], pd.NA).dropna()
    value = float(ratios.median()) if not ratios.empty else 0.0

    return {
        "kWp_effective": value,
        "kWp_effective_n_points": int(ratios.shape[0]),
        "kWp_effective_filters": {
            "poa_threshold_wm2": float(poa_threshold_wm2),
            "fit_mask_required": True,
        },
    }
