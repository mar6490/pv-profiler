from __future__ import annotations

import json
import time
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
        return index, str(index.tz)

    tz_to_use = timezone or "Etc/GMT-1"
    # Naive logger timestamps are interpreted as continuous local logger time.
    # Fixed-offset default avoids DST one-hour shifts in forward-model solar geometry.
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

    return out / day_key.map(denom)


def _rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def _bic(y: np.ndarray, yhat: np.ndarray, k: int) -> float:
    n = len(y)
    mse = float(np.mean((y - yhat) ** 2))
    mse = max(mse, 1e-12)
    return float(n * np.log(mse) + k * np.log(max(n, 2)))


def _minute_of_day(index: pd.DatetimeIndex) -> pd.Index:
    return pd.Index(index.hour * 60 + index.minute, name="minute_of_day")


def _poa_single(
    solar_position: pd.DataFrame,
    clearsky: pd.DataFrame,
    tilt: float,
    azimuth: float,
    dni_extra: pd.Series,
) -> pd.Series:
    return pvlib.irradiance.get_total_irradiance(
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


def _evaluate_candidate_samples(observed: np.ndarray, pred: pd.Series, k: int) -> tuple[float, float]:
    yhat = pred.to_numpy(dtype=float)
    mask = np.isfinite(observed) & np.isfinite(yhat)
    y = observed[mask]
    yh = yhat[mask]
    if len(y) == 0:
        return float("inf"), float("inf")
    return _rmse(y, yh), _bic(y, yh, k=k)


def _two_plane_azimuths(azimuth_center_deg: float, half_delta_az_deg: float) -> tuple[float, float]:
    az_east = (float(azimuth_center_deg) - float(half_delta_az_deg)) % 360
    az_west = (float(azimuth_center_deg) + float(half_delta_az_deg)) % 360
    return az_east, az_west


def _optimal_weight(observed: np.ndarray, p1_norm: np.ndarray, p2_norm: np.ndarray) -> float:
    a = p1_norm - p2_norm
    b = observed - p2_norm
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    denom = float(np.dot(a, a))
    if denom <= 0:
        return 0.5
    w = float(np.dot(a, b) / denom)
    return float(np.clip(w, 0.0, 1.0))


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
    two_plane_half_delta_az_deg: float = 90.0,
    skip_two_plane: bool = False,
    two_plane_if_rmse_ge: float = 0.0,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t_total0 = time.perf_counter()

    if not isinstance(p_norm_df.index, pd.DatetimeIndex):
        raise ValueError("Input p_norm dataframe must have DatetimeIndex")
    if "p_norm" not in p_norm_df.columns:
        raise ValueError("Input p_norm dataframe must contain 'p_norm' column")

    obs = p_norm_df[["p_norm"]].copy().sort_index()
    obs = obs.dropna(subset=["p_norm"])
    if obs.empty:
        raise ValueError("Input p_norm contains no usable rows")

    local_times, tz_used = _localize_logger_times(obs.index, timezone)
    obs_local = pd.Series(obs["p_norm"].to_numpy(dtype=float), index=local_times, name="p_norm")
    observed_arr = obs_local.to_numpy(dtype=float)

    t0 = time.perf_counter()
    location = pvlib.location.Location(latitude=latitude, longitude=longitude, tz=tz_used)
    solar_position = location.get_solarposition(local_times)
    clearsky = location.get_clearsky(local_times, model="ineichen")
    dni_extra = pvlib.irradiance.get_extra_radiation(local_times)
    t_precompute = time.perf_counter() - t0

    records: list[dict] = []

    tilts = np.arange(0, 61, tilt_step)
    azimuths = np.arange(60, 301, az_step)

    t0 = time.perf_counter()
    best_single: OrientationCandidate | None = None
    for tilt in tilts:
        for az in azimuths:
            poa = _poa_single(solar_position, clearsky, tilt=tilt, azimuth=az, dni_extra=dni_extra)
            p_hat = _daily_normalize(poa, quantile=quantile, norm_mode=norm_mode)
            rmse, bic = _evaluate_candidate_samples(observed_arr, p_hat, k=2)
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
            if best_single is None or cand.rmse < best_single.rmse:
                best_single = cand
    t_coarse_single = time.perf_counter() - t0

    assert best_single is not None

    t0 = time.perf_counter()
    t0s = int(round(best_single.params["tilt_deg"]))
    a0s = int(round(best_single.params["azimuth_deg"]))
    for tilt in np.arange(max(0, t0s - 5), min(60, t0s + 5) + 1, 1):
        for az in np.arange(max(60, a0s - 5), min(300, a0s + 5) + 1, 1):
            poa = _poa_single(solar_position, clearsky, tilt=tilt, azimuth=az, dni_extra=dni_extra)
            p_hat = _daily_normalize(poa, quantile=quantile, norm_mode=norm_mode)
            rmse, bic = _evaluate_candidate_samples(observed_arr, p_hat, k=2)
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
    t_fine_single = time.perf_counter() - t0

    run_two_plane = (not skip_two_plane) and (
        two_plane_if_rmse_ge <= 0 or best_single.rmse >= float(two_plane_if_rmse_ge)
    )

    t0 = time.perf_counter()
    best_two: OrientationCandidate | None = None
    if run_two_plane:
        centers = np.arange(60, 301, az_step)
        for tilt in tilts:
            for center in centers:
                az_e, az_w = _two_plane_azimuths(center, two_plane_half_delta_az_deg)
                poa_e = _poa_single(solar_position, clearsky, tilt=tilt, azimuth=az_e, dni_extra=dni_extra)
                poa_w = _poa_single(solar_position, clearsky, tilt=tilt, azimuth=az_w, dni_extra=dni_extra)

                p1_norm = _daily_normalize(poa_e, quantile=quantile, norm_mode=norm_mode)
                p2_norm = _daily_normalize(poa_w, quantile=quantile, norm_mode=norm_mode)

                w_opt = _optimal_weight(
                    observed_arr,
                    p1_norm.to_numpy(dtype=float),
                    p2_norm.to_numpy(dtype=float),
                )
                p_mix = w_opt * p1_norm + (1 - w_opt) * p2_norm

                rmse, bic = _evaluate_candidate_samples(observed_arr, p_mix, k=3)
                records.append(
                    {
                        "model_type": "two_plane",
                        "tilt_deg": float(tilt),
                        "azimuth_center_deg": float(center),
                        "two_plane_half_delta_az_deg": float(two_plane_half_delta_az_deg),
                        "azimuth_east_deg": az_e,
                        "azimuth_west_deg": az_w,
                        "weight_opt": float(w_opt),
                        "rmse": rmse,
                        "bic": bic,
                    }
                )
                cand = OrientationCandidate(
                    "two_plane",
                    {
                        "tilt_deg": float(tilt),
                        "azimuth_center_deg": float(center),
                        "two_plane_half_delta_az_deg": float(two_plane_half_delta_az_deg),
                        "azimuth_east_deg": az_e,
                        "azimuth_west_deg": az_w,
                        "weight_opt": float(w_opt),
                    },
                    rmse,
                    bic,
                )
                if best_two is None or cand.rmse < best_two.rmse:
                    best_two = cand
    t_coarse_two_plane = time.perf_counter() - t0

    choose_two = run_two_plane and best_two is not None and (best_two.rmse < best_single.rmse * 0.90) and (best_two.bic < best_single.bic)
    winner = best_two if choose_two else best_single

    timing = {
        "precompute": t_precompute,
        "coarse_single": t_coarse_single,
        "fine_single": t_fine_single,
        "coarse_two_plane": t_coarse_two_plane,
        "total": time.perf_counter() - t_total0,
    }

    result = {
        "model_type": winner.model_type,
        "tilt_deg": winner.params["tilt_deg"],
        "score_rmse": winner.rmse,
        "score_bic": winner.bic,
        "rmse_single": best_single.rmse,
        "bic_single": best_single.bic,
        "two_plane_run": bool(run_two_plane),
        "n_points": int(len(observed_arr)),
        "grid_spec": {
            "tilt_range": [0, 60],
            "tilt_step": int(tilt_step),
            "azimuth_range": [60, 300],
            "azimuth_step": int(az_step),
            "two_plane_half_delta_az_deg": float(two_plane_half_delta_az_deg),
            "two_plane_weight": "analytic_optimum",
        },
        "norm_mode": norm_mode,
        "quantile": quantile,
        "timezone_used": tz_used,
        "timing_seconds": timing,
    }
    if winner.model_type == "single":
        result["azimuth_deg"] = winner.params["azimuth_deg"]
    else:
        result.update(
            {
                "azimuth_center_deg": winner.params["azimuth_center_deg"],
                "two_plane_half_delta_az_deg": winner.params["two_plane_half_delta_az_deg"],
                "azimuth_east_deg": winner.params["azimuth_east_deg"],
                "azimuth_west_deg": winner.params["azimuth_west_deg"],
                "weight_east": winner.params["weight_opt"],
            }
        )

    top = pd.DataFrame(records).sort_values(["rmse", "bic"]).head(topk).reset_index(drop=True)

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
        p1_norm = _daily_normalize(poa_e, quantile=quantile, norm_mode=norm_mode)
        p2_norm = _daily_normalize(poa_w, quantile=quantile, norm_mode=norm_mode)
        pred = winner.params["weight_opt"] * p1_norm + (1 - winner.params["weight_opt"]) * p2_norm
    if winner.model_type == "single":
        pred = _daily_normalize(poa, quantile=quantile, norm_mode=norm_mode)
    pred_prof = pred.groupby(_minute_of_day(pred.index)).median().rename("predicted_p_norm")
    minutes = pd.Index(np.arange(0, 24 * 60, 5), name="minute_of_day")
    profile_compare = pd.concat([obs_prof, pred_prof], axis=1).reindex(minutes).reset_index()

    all_records = pd.DataFrame(records)
    single_full = (
        all_records.loc[all_records["model_type"] == "single", ["tilt_deg", "azimuth_deg", "rmse", "bic"]]
        .drop_duplicates(subset=["tilt_deg", "azimuth_deg"], keep="first")
        .sort_values(["tilt_deg", "azimuth_deg"])
        .reset_index(drop=True)
    )
    two_plane_cols = ["tilt_deg", "azimuth_center_deg", "weight_opt", "rmse", "bic"]
    two_plane_subset = all_records.loc[all_records["model_type"] == "two_plane"]
    if two_plane_subset.empty:
        two_plane_full = pd.DataFrame(columns=two_plane_cols)
    else:
        two_plane_full = (
            two_plane_subset[two_plane_cols]
            .drop_duplicates(subset=["tilt_deg", "azimuth_center_deg", "weight_opt"], keep="first")
            .sort_values(["tilt_deg", "azimuth_center_deg", "weight_opt"])
            .reset_index(drop=True)
        )

    return result, top, profile_compare, single_full, two_plane_full


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
    two_plane_delta_az_deg: float = 90.0,
    skip_two_plane: bool = False,
    two_plane_if_rmse_ge: float = 0.0,
) -> dict:
    df = pd.read_parquet(input_p_norm_parquet)
    result, topk_df, profile_compare, single_full, two_plane_full = run_block5_orientation_fit(
        df,
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
        tilt_step=tilt_step,
        az_step=az_step,
        topk=topk,
        quantile=quantile,
        norm_mode=norm_mode,
        two_plane_half_delta_az_deg=two_plane_delta_az_deg,
        skip_two_plane=skip_two_plane,
        two_plane_if_rmse_ge=two_plane_if_rmse_ge,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "08_orientation_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    topk_df.to_csv(out / "09_orientation_topk.csv", index=False)
    single_full.to_csv(out / "09a_orientation_single_full_grid.csv", index=False)
    two_plane_full.to_csv(out / "09b_orientation_two_plane_full_grid.csv", index=False)
    profile_compare.to_csv(out / "10_profile_compare.csv", index=False)
    return result
