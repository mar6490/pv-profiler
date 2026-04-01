from __future__ import annotations

import pandas as pd

from .types import DiagnosticsResult


def compute_diagnostics(series: pd.Series, extra_warnings: list[str] | None = None) -> DiagnosticsResult:
    inferred = pd.infer_freq(series.index)
    n_samples = int(series.shape[0])
    coverage = float(series.notna().mean()) if n_samples else 0.0
    warnings = list(extra_warnings or [])

    if n_samples == 0:
        warnings.append("Series is empty after preprocessing.")
    if coverage < 0.95:
        warnings.append(f"Coverage below 95% ({coverage:.1%}).")

    return DiagnosticsResult(
        n_samples=n_samples,
        coverage_fraction=coverage,
        frequency=inferred,
        warnings=warnings,
    )
