from __future__ import annotations

import pandas as pd


def run_sdt_onboarding(power_series: pd.Series) -> tuple[pd.Series, list[str]]:
    """Run a lightweight SDT onboarding flow.

    Returns cleaned/reindexed series and warning messages.
    """
    warnings: list[str] = []
    cleaned = power_series.copy().sort_index()

    inferred = pd.infer_freq(cleaned.index)
    if inferred is None:
        inferred = "5min"
        warnings.append("Could not infer frequency; using 5min default for regularization.")

    cleaned = cleaned.resample(inferred).mean().interpolate(limit=2)

    try:
        from solardatatools import DataHandler  # type: ignore

        # SDT expects a regularized daytime-like signal. We run a minimal pipeline
        # and then keep the cleaned signal returned by SDT if available.
        df = cleaned.to_frame(name="power")
        handler = DataHandler(data_frame=df)
        handler.run_pipeline(power_col="power", fix_shifts=True, verbose=False)
        if hasattr(handler, "filled_data_matrix") and handler.filled_data_matrix is not None:
            series = pd.Series(handler.data_frame_raw["power"].values, index=handler.data_frame_raw.index)
            cleaned = series.sort_index()
    except Exception as exc:
        warnings.append(f"SDT onboarding fallback used ({exc.__class__.__name__}: {exc}).")

    return cleaned, warnings
