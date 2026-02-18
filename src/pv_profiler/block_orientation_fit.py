from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib


@dataclass
class OrientationCandidate:
    model_type: str
    params: dict[str, float]
    rmse: float
    bic: float


def _localize_logger_times(index: pd.DatetimeIndex, timezone: str | None) -> tuple[pd.DatetimeIndex, str]:
    if index.tz is not None:
        # If timestamps are already tz-aware we trust the provided zone from data.
        return index, str(index.tz)

    tz_to_use = timezone or "Etc/GMT-1"
    # Naive logger timestamps are interpreted as continuous local logger time.
    # Default is fixed-offset zone to avoid DST-induced one-hour jumps.
    return index.tz_localize(tz_to_use), tz_to_use


def _daily_normalize(poa: pd.Series, quantile: float, norm_mode: str) -> pd.Series:
    out = poa.copy()
    day_key = out.index.normalize()

    if norm_mode == "quantile":
        denom = out.groupby(day_key).quantile(quantile)
    elif norm_mode == "max":
        denom = out.groupby(day_key).max()
    else:
        raise ValueError("norm_mode must be one of: quantile, max")

    fallback_max = out.groupby(day_key).max()
    denom = denom.where((denom > 0) & denom.notna(), fallback_max)
    denom = denom.where((denom > 0) & denom.notna())

    mapped = day_key.map(denom)
    out = out / mapped
    return out


def _rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def _bic(y: np.ndarray, yhat: np.ndarray, k: int) -> float:
    n = len(y)
    mse = float(np.mean((y - yhat) ** 2))
    mse = max(mse, 1e-12)
    return float(n * np.log(mse) + k * np.log(max(n, 2)))


def _minute_of_day(index: pd.DatetimeIndex) -> pd.Index:
    return pd.Index(index.hour * 60 + index.minute, name="minute_of_day")


def _prepare_target(series: pd.Series, fit_target: str) -> tuple[np.ndarray, pd.Index]:
    minute = _minute_of_day(series.index)
    if fit_target == "median":
        grouped = series.groupby(minute).median().dropna()
        return grouped.to_numpy(dtype=float), grouped.index
    if fit_target == "samples":
        return series.to_numpy(dtype=float), series.index
    raise ValueError("fit_target must be one of: median, samples")


def _prepare_prediction(pred_series: pd.Series, fit_target: str, ref_index: pd.Index) -> np.ndarray:
    if fit_target == "median":
        minute = _minute_of_day(pred_series.index)
        grouped = pred_series.groupby(minute).median()
        grouped = grouped.reindex(ref_index)
        return grouped.to_numpy(dtype=float)
    arr = pred_series.reindex(ref_index).to_numpy(dtype=float)
    return arr


def _poa_single(
    solar_position: pd.DataFrame,
    clearsky: pd.DataFrame,
    tilt: float,
    azimuth: float,
    dni_extra: pd.Series,
) -> pd.Series:
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=float(tilt),
        surface_azimuth=float(azimuth),
        solar_zenith=solar_position["apparent_zenith"],
        solar_azimuth=solar_position["azimuth"],
        dni=clearsky["dni"],
        ghi=clearsky["ghi"],
        dhi=clearsky["dhi"],
        dni_extra=dni_extra,
        model="haydavies",
    )["poa_global"].clip(lower=0)
    return poa


def _evaluate_candidate(
    observed: pd.Series,
    pred: pd.Series,
    fit_target: str,
    k: int,
    ref_index: pd.Index,
) -> tuple[float, float]:
    y = _prepare_target(observed, fit_target)[0]
    yhat = _prepare_prediction(pred, fit_target, ref_index)
    mask = np.isfinite(y) & np.isfinite(yhat)
    y = y[mask]
    yhat = yhat[mask]
    if len(y) == 0:
        return float("inf"), float("inf")
    return _rmse(y, yhat), _bic(y, yhat, k=k)


def run_block5_orientation_fit(
    p_norm_df: pd.DataFrame,
    *,
    latitude: float,
    longitude: float,
    timezone: str | None = None,
    tilt_step: int = 5,
    az_step: int = 10,
    topk: int = 20,
    quantile: float = 0.995,
    norm_mode: str = "quantile",
    fit_target: str = "median",
) -> tuple[dict, pd.DataFrame, pd.DataFrame | None]:
    if not isinstance(p_norm_df.index, pd.DatetimeIndex):
        raise ValueError("Input p_norm dataframe must have DatetimeIndex")
    if "p_norm" not in p_norm_df.columns:
        raise ValueError("Input p_norm dataframe must contain 'p_norm' column")

    obs = p_norm_df[["p_norm"]].copy().sort_index()
    obs = obs.dropna(subset=["p_norm"])
    if obs.empty:
        raise ValueError("Input p_norm contains no usable rows")

    local_times, tz_used = _localize_logger_times(obs.index, timezone)
    obs_local = pd.Series(obs["p_norm"].to_numpy(), index=local_times, name="p_norm")

    location = pvlib.location.Location(latitude=latitude, longitude=longitude, tz=tz_used)
    solar_position = location.get_solarposition(local_times)
    clearsky = location.get_clearsky(local_times, model="ineichen")
    dni_extra = pvlib.irradiance.get_extra_radiation(local_times)

    y_ref, ref_index = _prepare_target(obs_local, fit_target)

    records: list[dict] = []

    tilts = np.arange(0, 61, tilt_step)
    azimuths = np.arange(60, 301, az_step)

    best_single: OrientationCandidate | None = None
    for tilt in tilts:
        for az in azimuths:
            poa = _poa_single(solar_position, clearsky, tilt=tilt, azimuth=az, dni_extra=dni_extra)
            p_hat = _daily_normalize(poa, quantile=quantile, norm_mode=norm_mode)
            rmse, bic = _evaluate_candidate(obs_local, p_hat, fit_target, k=2, ref_index=ref_index)
            row = {
                "model_type": "single",
                "tilt_deg": float(tilt),
                "azimuth_deg": float(az),
                "rmse": rmse,
                "bic": bic,
            }
            records.append(row)
            cand = OrientationCandidate("single", {"tilt_deg": float(tilt), "azimuth_deg": float(az)}, rmse, bic)
            if best_single is None or cand.rmse < best_single.rmse:
                best_single = cand

    assert best_single is not None

    # fine grid around best single
    t0 = int(round(best_single.params["tilt_deg"]))
    a0 = int(round(best_single.params["azimuth_deg"]))
    for tilt in np.arange(max(0, t0 - 5), min(60, t0 + 5) + 1, 1):
        for az in np.arange(max(60, a0 - 5), min(300, a0 + 5) + 1, 1):
            poa = _poa_single(solar_position, clearsky, tilt=tilt, azimuth=az, dni_extra=dni_extra)
            p_hat = _daily_normalize(poa, quantile=quantile, norm_mode=norm_mode)
            rmse, bic = _evaluate_candidate(obs_local, p_hat, fit_target, k=2, ref_index=ref_index)
            records.append(
                {
                    "model_type": "single",
                    "tilt_deg": float(tilt),
                    "azimuth_deg": float(az),
                    "rmse": rmse,
                    "bic": bic,
                }
            )
            cand = OrientationCandidate("single", {"tilt_deg": float(tilt), "azimuth_deg": float(az)}, rmse, bic)
            if cand.rmse < best_single.rmse:
                best_single = cand

    best_two: OrientationCandidate | None = None
    centers = np.arange(60, 301, az_step)
    weights = np.arange(0, 1.0001, 0.05)

    for tilt in tilts:
        for center in centers:
            az_e = float(max(0, center - 45))
            az_w = float(min(360, center + 45))
            poa_e = _poa_single(solar_position, clearsky, tilt=tilt, azimuth=az_e, dni_extra=dni_extra)
            poa_w = _poa_single(solar_position, clearsky, tilt=tilt, azimuth=az_w, dni_extra=dni_extra)
            for w in weights:
                poa_mix = float(w) * poa_e + float(1 - w) * poa_w
                p_hat = _daily_normalize(poa_mix, quantile=quantile, norm_mode=norm_mode)
                rmse, bic = _evaluate_candidate(obs_local, p_hat, fit_target, k=3, ref_index=ref_index)
                records.append(
                    {
                        "model_type": "two_plane",
                        "tilt_deg": float(tilt),
                        "azimuth_center_deg": float(center),
                        "delta_az_deg": 90.0,
                        "azimuth_east_deg": az_e,
                        "azimuth_west_deg": az_w,
                        "weight_east": float(w),
                        "rmse": rmse,
                        "bic": bic,
                    }
                )
                cand = OrientationCandidate(
                    "two_plane",
                    {
                        "tilt_deg": float(tilt),
                        "azimuth_center_deg": float(center),
                        "delta_az_deg": 90.0,
                        "azimuth_east_deg": az_e,
                        "azimuth_west_deg": az_w,
                        "weight_east": float(w),
                    },
                    rmse,
                    bic,
                )
                if best_two is None or cand.rmse < best_two.rmse:
                    best_two = cand

    assert best_two is not None

    choose_two = (best_two.rmse < best_single.rmse * 0.90) and (best_two.bic < best_single.bic)
    winner = best_two if choose_two else best_single

    result = {
        "model_type": winner.model_type,
        "tilt_deg": winner.params["tilt_deg"],
        "score_rmse": winner.rmse,
        "score_bic": winner.bic,
        "n_points": int(len(y_ref)),
        "grid_spec": {
            "tilt_range": [0, 60],
            "tilt_step": int(tilt_step),
            "azimuth_range": [60, 300],
            "azimuth_step": int(az_step),
            "two_plane_delta_az_deg": 90,
            "weight_east_step": 0.05,
        },
        "norm_mode": norm_mode,
        "quantile": quantile,
        "fit_target": fit_target,
        "timezone_used": tz_used,
    }
    if winner.model_type == "single":
        result["azimuth_deg"] = winner.params["azimuth_deg"]
    else:
        result.update(
            {
                "azimuth_center_deg": winner.params["azimuth_center_deg"],
                "delta_az_deg": winner.params["delta_az_deg"],
                "azimuth_east_deg": winner.params["azimuth_east_deg"],
                "azimuth_west_deg": winner.params["azimuth_west_deg"],
                "weight_east": winner.params["weight_east"],
            }
        )

    top = pd.DataFrame(records).sort_values(["rmse", "bic"]).head(topk).reset_index(drop=True)

    profile_compare: pd.DataFrame | None = None
    if fit_target == "median":
        obs_prof = obs_local.groupby(_minute_of_day(obs_local.index)).median().rename("observed_p_norm")
        if winner.model_type == "single":
            poa = _poa_single(
                solar_position,
                clearsky,
                tilt=winner.params["tilt_deg"],
                azimuth=winner.params["azimuth_deg"],
                dni_extra=dni_extra,
            )
        else:
            poa_e = _poa_single(
                solar_position,
                clearsky,
                tilt=winner.params["tilt_deg"],
                azimuth=winner.params["azimuth_east_deg"],
                dni_extra=dni_extra,
            )
            poa_w = _poa_single(
                solar_position,
                clearsky,
                tilt=winner.params["tilt_deg"],
                azimuth=winner.params["azimuth_west_deg"],
                dni_extra=dni_extra,
            )
            poa = winner.params["weight_east"] * poa_e + (1 - winner.params["weight_east"]) * poa_w
        pred = _daily_normalize(poa, quantile=quantile, norm_mode=norm_mode)
        pred_prof = pred.groupby(_minute_of_day(pred.index)).median().rename("predicted_p_norm")
        profile_compare = pd.concat([obs_prof, pred_prof], axis=1).reset_index()

    return result, top, profile_compare


def run_block5_from_files(
    input_p_norm_parquet: str | Path,
    output_dir: str | Path,
    *,
    latitude: float,
    longitude: float,
    timezone: str | None = None,
    tilt_step: int = 5,
    az_step: int = 10,
    topk: int = 20,
    quantile: float = 0.995,
    norm_mode: str = "quantile",
    fit_target: str = "median",
) -> dict:
    df = pd.read_parquet(input_p_norm_parquet)
    result, topk_df, profile_compare = run_block5_orientation_fit(
        df,
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
        tilt_step=tilt_step,
        az_step=az_step,
        topk=topk,
        quantile=quantile,
        norm_mode=norm_mode,
        fit_target=fit_target,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "08_orientation_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    topk_df.to_csv(out / "09_orientation_topk.csv", index=False)
    if profile_compare is not None:
        profile_compare.to_csv(out / "10_profile_compare.csv", index=False)
    return result
