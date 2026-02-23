from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pvlib

from pv_profiler.block_orientation_fit import _optimal_weight, _two_plane_azimuths, run_block5_from_files


def _daily_quantile_normalize(series: pd.Series, q: float = 0.995) -> pd.Series:
    day = series.index.normalize()
    denom = series.groupby(day).quantile(q)
    fallback = series.groupby(day).max()
    denom = denom.where((denom > 0) & denom.notna(), fallback)
    denom = denom.where((denom > 0) & denom.notna())
    return series / day.map(denom)


def _make_synthetic_single(latitude: float, longitude: float, tilt: float, azimuth: float) -> pd.DataFrame:
    times = pd.date_range("2020-04-01", periods=45 * 24, freq="h", tz="Etc/GMT-1")
    loc = pvlib.location.Location(latitude=latitude, longitude=longitude, tz="Etc/GMT-1")
    sp = loc.get_solarposition(times)
    cs = loc.get_clearsky(times, model="ineichen")
    dni_extra = pvlib.irradiance.get_extra_radiation(times)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=sp["apparent_zenith"],
        solar_azimuth=sp["azimuth"],
        dni=cs["dni"],
        ghi=cs["ghi"],
        dhi=cs["dhi"],
        dni_extra=dni_extra,
        model="haydavies",
    )["poa_global"].clip(lower=0)
    p_norm = _daily_quantile_normalize(poa)
    return pd.DataFrame({"p_norm": p_norm.to_numpy()}, index=times.tz_localize(None))


def _make_synthetic_two_plane(latitude: float, longitude: float, tilt: float, center: float, w: float) -> pd.DataFrame:
    times = pd.date_range("2020-04-01", periods=45 * 24, freq="h", tz="Etc/GMT-1")
    loc = pvlib.location.Location(latitude=latitude, longitude=longitude, tz="Etc/GMT-1")
    sp = loc.get_solarposition(times)
    cs = loc.get_clearsky(times, model="ineichen")
    dni_extra = pvlib.irradiance.get_extra_radiation(times)

    az_e = center - 45
    az_w = center + 45
    poa_e = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=az_e,
        solar_zenith=sp["apparent_zenith"],
        solar_azimuth=sp["azimuth"],
        dni=cs["dni"],
        ghi=cs["ghi"],
        dhi=cs["dhi"],
        dni_extra=dni_extra,
        model="haydavies",
    )["poa_global"].clip(lower=0)
    poa_w = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=az_w,
        solar_zenith=sp["apparent_zenith"],
        solar_azimuth=sp["azimuth"],
        dni=cs["dni"],
        ghi=cs["ghi"],
        dhi=cs["dhi"],
        dni_extra=dni_extra,
        model="haydavies",
    )["poa_global"].clip(lower=0)

    mix = w * poa_e + (1 - w) * poa_w
    p_norm = _daily_quantile_normalize(mix)
    return pd.DataFrame({"p_norm": p_norm.to_numpy()}, index=times.tz_localize(None))


def test_block5_single_plane_recovery(tmp_path):
    lat, lon = 52.45544, 13.52481
    df = _make_synthetic_single(lat, lon, tilt=25, azimuth=200)
    input_parquet = tmp_path / "07_p_norm_clear.parquet"
    df.to_parquet(input_parquet)

    out_dir = tmp_path / "out"
    result = run_block5_from_files(
        input_p_norm_parquet=input_parquet,
        output_dir=out_dir,
        latitude=lat,
        longitude=lon,
        tilt_step=20,
        az_step=20,
        topk=5,
    )

    assert result["model_type"] == "single"
    assert abs(float(result["tilt_deg"]) - 25) <= 5
    assert abs(float(result["azimuth_deg"]) - 200) <= 10
    assert "timing_seconds" in result
    profile = pd.read_csv(out_dir / "10_profile_compare.csv")
    assert {"minute_of_day", "observed_p_norm", "predicted_p_norm"}.issubset(profile.columns)
    single_full = pd.read_csv(out_dir / "09a_orientation_single_full_grid.csv")
    assert {"tilt_deg", "azimuth_deg", "rmse", "bic"}.issubset(single_full.columns)


def test_block5_two_plane_selected(tmp_path):
    lat, lon = 52.45544, 13.52481
    df = _make_synthetic_two_plane(lat, lon, tilt=20, center=180, w=0.5)
    input_parquet = tmp_path / "07_p_norm_clear.parquet"
    df.to_parquet(input_parquet)

    out_dir = tmp_path / "out"
    result = run_block5_from_files(
        input_p_norm_parquet=input_parquet,
        output_dir=out_dir,
        latitude=lat,
        longitude=lon,
        tilt_step=20,
        az_step=20,
        topk=5,
    )

    assert result["two_plane_run"] is True
    profile = pd.read_csv(out_dir / "10_profile_compare.csv")
    assert {"minute_of_day", "observed_p_norm", "predicted_p_norm"}.issubset(profile.columns)
    assert (out_dir / "09b_orientation_two_plane_full_grid.csv").exists()


def test_plot_script_generates_expected_files(tmp_path):
    lat, lon = 52.45544, 13.52481
    df = _make_synthetic_single(lat, lon, tilt=25, azimuth=200)
    out_dir = tmp_path / "out"
    input_parquet = tmp_path / "07_p_norm_clear.parquet"
    df.to_parquet(input_parquet)

    run_block5_from_files(
        input_p_norm_parquet=input_parquet,
        output_dir=out_dir,
        latitude=lat,
        longitude=lon,
        tilt_step=20,
        az_step=20,
        topk=5,
    )

    cmd = [
        sys.executable,
        "scripts/plot_block5_results.py",
        "--input-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)

    assert (out_dir / "plot_rmse_heatmap.png").exists()
    assert (out_dir / "plot_profile_compare.png").exists()
    assert (out_dir / "plot_residual_vs_time.png").exists()
    assert (out_dir / "plot_rmse_vs_azimuth.png").exists()
    assert (out_dir / "block5_diagnostics.pdf").exists()


def test_two_plane_azimuth_half_delta_semantics():
    az_e, az_w = _two_plane_azimuths(180, 90)
    assert ((az_w - az_e) % 360) == 180
    assert az_e == 90
    assert az_w == 270


def test_analytic_weight_recovers_true_weight_and_beats_coarse_grid():
    rng = np.random.default_rng(0)
    p1 = rng.uniform(0.2, 1.2, 200)
    p2 = rng.uniform(0.1, 1.0, 200)
    w_true = 0.35
    y = w_true * p1 + (1 - w_true) * p2

    w_opt = _optimal_weight(y, p1, p2)
    assert abs(w_opt - w_true) < 0.05

    rmse_opt = float(np.sqrt(np.mean((y - (w_opt * p1 + (1 - w_opt) * p2)) ** 2)))
    coarse = [0.0, 0.5, 1.0]
    rmse_grid = min(float(np.sqrt(np.mean((y - (w * p1 + (1 - w) * p2)) ** 2))) for w in coarse)
    assert rmse_opt <= rmse_grid + 1e-12


def test_two_plane_skip_threshold_sets_flag(tmp_path):
    lat, lon = 52.45544, 13.52481
    df = _make_synthetic_single(lat, lon, tilt=25, azimuth=200)
    input_parquet = tmp_path / "07_p_norm_clear.parquet"
    df.to_parquet(input_parquet)

    out_dir = tmp_path / "out"
    result = run_block5_from_files(
        input_p_norm_parquet=input_parquet,
        output_dir=out_dir,
        latitude=lat,
        longitude=lon,
        tilt_step=20,
        az_step=20,
        topk=5,
        two_plane_if_rmse_ge=1.0,
    )

    assert result["two_plane_run"] is False
