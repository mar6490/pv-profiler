"""Block D: orientation model fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pv_profiler.validation import INTERNAL_TZ


@dataclass
class OrientationFitArtifacts:
    """Container for best-fit orientation outputs."""

    result: dict[str, Any]
    diagnostics: pd.DataFrame
    poa_unshaded: pd.Series


def _normalize_daily_q995(series: pd.Series) -> pd.Series:
    dates = pd.Index(series.index.tz_convert(INTERNAL_TZ).date)
    q = series.groupby(dates).transform(lambda x: x.quantile(0.995))
    return series / q.replace(0.0, np.nan)


def _compute_loss(obs: pd.Series, pred: pd.Series, mode: str) -> float:
    residual = (obs - pred).dropna()
    if residual.empty:
        return float("inf")
    if mode == "pooled_rmse":
        return float(np.sqrt(np.mean(np.square(residual.values))))

    dates = pd.Index(residual.index.tz_convert(INTERNAL_TZ).date)
    daily_rmse = residual.groupby(dates).apply(lambda x: float(np.sqrt(np.mean(np.square(x.values)))))
    return float(daily_rmse.median())


def _fit_scale(y: np.ndarray, x: np.ndarray) -> float:
    denom = float(np.dot(x, x))
    if denom <= 0:
        return 0.0
    return float(np.dot(y, x) / denom)


def _poa_for_orientation(
    times: pd.DatetimeIndex,
    lat: float,
    lon: float,
    tilt: float,
    azimuth: float,
    transposition_model: str,
) -> pd.Series:
    import pvlib

    location = pvlib.location.Location(latitude=lat, longitude=lon, tz=INTERNAL_TZ)
    solar = location.get_solarposition(times)
    clearsky = location.get_clearsky(times, model="ineichen")
    dni_extra = None
    get_extra = getattr(pvlib.irradiance, "get_extra_radiation", None)
    if callable(get_extra):
        dni_extra = get_extra(times)
    kwargs = dict(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=solar["apparent_zenith"],
        solar_azimuth=solar["azimuth"],
        dni=clearsky["dni"],
        ghi=clearsky["ghi"],
        dhi=clearsky["dhi"],
        model=transposition_model,
    )
    if dni_extra is not None:
        kwargs["dni_extra"] = dni_extra
    try:
        poa = pvlib.irradiance.get_total_irradiance(**kwargs)["poa_global"]
    except TypeError:
        kwargs.pop("dni_extra", None)
        poa = pvlib.irradiance.get_total_irradiance(**kwargs)["poa_global"]
    return poa.rename("poa_global")


def fit_orientation(
    ac_power_clean: pd.Series,
    fit_mask: pd.Series,
    *,
    lat: float,
    lon: float,
    config: dict[str, Any],
) -> OrientationFitArtifacts:
    """Fit single-plane vs east-west two-plane orientation model."""
    cfg = config.get("orientation", {})
    az_min = int(cfg.get("az_min", 90))
    az_max = int(cfg.get("az_max", 270))
    coarse_tilt_step = int(cfg.get("coarse_tilt_step", 5))
    coarse_az_step = int(cfg.get("coarse_az_step", 10))
    refine_tilt_step = int(cfg.get("refine_tilt_step", 1))
    refine_az_step = int(cfg.get("refine_az_step", 2))
    enable_two_plane = bool(cfg.get("enable_two_plane", True))
    ew_improve_threshold = float(cfg.get("ew_improve_threshold", 0.10))
    top_n = int(cfg.get("top_n", 50))
    transposition_model = str(cfg.get("transposition_model", "perez"))
    loss_mode = str(cfg.get("loss_mode", "median_daily_rmse"))

    mask = fit_mask.reindex(ac_power_clean.index).fillna(False).astype(bool)
    obs_raw = ac_power_clean.astype(float)
    obs_norm = _normalize_daily_q995(obs_raw)
    times = ac_power_clean.index
    max_eval_points = int(cfg.get("max_eval_points", 3000))

    eval_mask = mask.copy()
    n_mask = int(eval_mask.sum())
    if n_mask > max_eval_points > 0:
        stride = int(np.ceil(n_mask / max_eval_points))
        selected = np.flatnonzero(eval_mask.to_numpy())[::stride]
        eval_mask = pd.Series(False, index=mask.index)
        eval_mask.iloc[selected] = True

    eval_times = times[eval_mask]
    eval_obs_raw = obs_raw[eval_mask]
    eval_obs_norm = obs_norm[eval_mask]

    candidates: list[dict[str, Any]] = []
    best_single: dict[str, Any] | None = None

    def evaluate_single(tilt: float, azimuth: float) -> dict[str, Any]:
        poa = _poa_for_orientation(eval_times, lat, lon, tilt, azimuth, transposition_model)
        sel = poa.notna() & eval_obs_raw.notna()
        s = _fit_scale(eval_obs_raw[sel].values, poa[sel].values) if sel.any() else 0.0
        pred_norm = _normalize_daily_q995((s * poa).rename("pred"))
        score = _compute_loss(eval_obs_norm, pred_norm.reindex(eval_obs_norm.index), loss_mode)
        return {
            "model_type": "single",
            "tilt_deg": float(tilt),
            "azimuth_deg": float(azimuth),
            "score_rmse": float(score),
            "scale": float(s),
        }

    for tilt in np.arange(0, 60 + coarse_tilt_step, coarse_tilt_step):
        for azimuth in np.arange(az_min, az_max + coarse_az_step, coarse_az_step):
            result = evaluate_single(float(tilt), float(azimuth))
            candidates.append(dict(result))
            if best_single is None or result["score_rmse"] < best_single["score_rmse"]:
                best_single = result

    assert best_single is not None
    t0 = best_single["tilt_deg"]
    a0 = best_single["azimuth_deg"]

    for tilt in np.arange(max(0, t0 - 5), min(60, t0 + 5) + refine_tilt_step, refine_tilt_step):
        for azimuth in np.arange(max(az_min, a0 - 10), min(az_max, a0 + 10) + refine_az_step, refine_az_step):
            result = evaluate_single(float(tilt), float(azimuth))
            candidates.append(dict(result))
            if result["score_rmse"] < best_single["score_rmse"]:
                best_single = result

    best_two: dict[str, Any] | None = None
    if enable_two_plane:
        for tilt in np.arange(0, 60 + coarse_tilt_step, coarse_tilt_step):
            poa_e = _poa_for_orientation(eval_times, lat, lon, float(tilt), 90.0, transposition_model)
            poa_w = _poa_for_orientation(eval_times, lat, lon, float(tilt), 270.0, transposition_model)
            sel = poa_e.notna() & poa_w.notna() & eval_obs_raw.notna()
            if not sel.any():
                continue
            X = np.column_stack([poa_e[sel].values, poa_w[sel].values])
            y = eval_obs_raw[sel].values
            coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
            s_e, s_w = [float(max(0.0, x)) for x in coeffs]
            pred_raw = s_e * poa_e + s_w * poa_w
            pred_norm = _normalize_daily_q995(pred_raw)
            score = _compute_loss(eval_obs_norm, pred_norm.reindex(eval_obs_norm.index), loss_mode)
            item = {
                "model_type": "two-plane",
                "tilt_deg": float(tilt),
                "azimuth_deg_e": 90.0,
                "azimuth_deg_w": 270.0,
                "score_rmse": float(score),
                "sE": s_e,
                "sW": s_w,
                "ew_ratio": float(s_e / s_w) if s_w > 0 else np.nan,
                "poa": poa_e + poa_w,
            }
            candidates.append({k: v for k, v in item.items() if k != "poa"})
            if best_two is None or item["score_rmse"] < best_two["score_rmse"]:
                best_two = item

    use_two_plane = False
    if enable_two_plane and best_two is not None and best_single["score_rmse"] > 0:
        improvement = (best_single["score_rmse"] - best_two["score_rmse"]) / best_single["score_rmse"]
        use_two_plane = improvement >= ew_improve_threshold

    best = best_two if use_two_plane and best_two is not None else best_single
    assert best is not None

    fit_idx = mask & obs_raw.notna()
    n_fit_points = int(fit_idx.sum())
    n_fit_days = int(pd.Index(obs_raw.index[fit_idx].tz_convert(INTERNAL_TZ).date).nunique()) if n_fit_points else 0

    result = {
        "model_type": best["model_type"],
        "tilt_deg": float(best["tilt_deg"]),
        "score_rmse": float(best["score_rmse"]),
        "n_fit_points": n_fit_points,
        "n_fit_days": n_fit_days,
        "n_eval_points": int(eval_mask.sum()),
        "grid_coarse_step": {"tilt": coarse_tilt_step, "azimuth": coarse_az_step},
        "grid_refine_step": {"tilt": refine_tilt_step, "azimuth": refine_az_step},
        "az_min": az_min,
        "az_max": az_max,
        "transposition_model": transposition_model,
        "loss_mode": loss_mode,
    }
    if best["model_type"] == "single":
        result["azimuth_deg"] = float(best["azimuth_deg"])
    else:
        result["azimuth_deg_e"] = 90.0
        result["azimuth_deg_w"] = 270.0
        result["sE"] = float(best["sE"])
        result["sW"] = float(best["sW"])
        result["ew_ratio"] = float(best["ew_ratio"]) if np.isfinite(best["ew_ratio"]) else None

    if best["model_type"] == "single":
        best_poa_full = _poa_for_orientation(times, lat, lon, float(best["tilt_deg"]), float(best["azimuth_deg"]), transposition_model)
    else:
        poa_e_full = _poa_for_orientation(times, lat, lon, float(best["tilt_deg"]), 90.0, transposition_model)
        poa_w_full = _poa_for_orientation(times, lat, lon, float(best["tilt_deg"]), 270.0, transposition_model)
        best_poa_full = float(best.get("sE", 0.0)) * poa_e_full + float(best.get("sW", 0.0)) * poa_w_full

    diagnostics = pd.DataFrame(candidates)
    if not diagnostics.empty:
        diagnostics = diagnostics.sort_values("score_rmse", ascending=True).head(top_n)

    return OrientationFitArtifacts(result=result, diagnostics=diagnostics, poa_unshaded=best_poa_full.rename("poa_cs"))
