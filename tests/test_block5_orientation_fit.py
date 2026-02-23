from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pvlib

from pv_profiler.block_orientation_fit import _optimal_weight, _two_plane_azimuths, run_block5_from_files, run_block5_orientation_fit


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
    airmass = pvlib.atmosphere.get_relative_airmass(sp["apparent_zenith"])
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=sp["apparent_zenith"],
        solar_azimuth=sp["azimuth"],
        dni=cs["dni"],
        ghi=cs["ghi"],
        dhi=cs["dhi"],
        dni_extra=dni_extra,
        airmass=airmass,
        model="perez",
    )["poa_global"].clip(lower=0)
    p_norm = _daily_quantile_normalize(poa)
    return pd.DataFrame({"p_norm": p_norm.to_numpy()}, index=times.tz_localize(None))


def _make_synthetic_two_plane(latitude: float, longitude: float, tilt: float, center: float, w: float) -> pd.DataFrame:
    times = pd.date_range("2020-04-01", periods=45 * 24, freq="h", tz="Etc/GMT-1")
    loc = pvlib.location.Location(latitude=latitude, longitude=longitude, tz="Etc/GMT-1")
    sp = loc.get_solarposition(times)
    cs = loc.get_clearsky(times, model="ineichen")
    dni_extra = pvlib.irradiance.get_extra_radiation(times)
    airmass = pvlib.atmosphere.get_relative_airmass(sp["apparent_zenith"])

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
        airmass=airmass,
        model="perez",
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
        airmass=airmass,
        model="perez",
    )["poa_global"].clip(lower=0)

    mix = w * poa_e + (1 - w) * poa_w
    p_norm = _daily_quantile_normalize(mix)
    return pd.DataFrame({"p_norm": p_norm.to_numpy()}, index=times.tz_localize(None))


def test_poa_single_uses_perez_with_airmass(monkeypatch):
    from pv_profiler.block_orientation_fit import _poa_single

    idx = pd.date_range("2020-01-01", periods=3, freq="h", tz="Etc/GMT-1")
    solar_position = pd.DataFrame(
        {"apparent_zenith": [80.0, 60.0, 70.0], "azimuth": [100.0, 120.0, 140.0]},
        index=idx,
    )
    clearsky = pd.DataFrame({"dni": [0.0, 400.0, 100.0], "ghi": [0.0, 300.0, 100.0], "dhi": [0.0, 80.0, 20.0]}, index=idx)
    dni_extra = pd.Series([1367.0, 1367.0, 1367.0], index=idx)

    seen: dict[str, object] = {}

    def fake_get_total_irradiance(**kwargs):
        seen.update(kwargs)
        return {"poa_global": pd.Series([0.1, 0.2, 0.1], index=idx)}

    monkeypatch.setattr("pv_profiler.block_orientation_fit.pvlib.irradiance.get_total_irradiance", fake_get_total_irradiance)

    _poa_single(solar_position, clearsky, tilt=20, azimuth=180, dni_extra=dni_extra)

    assert seen["model"] == "perez"
    assert seen["airmass"] is not None


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


def test_two_plane_fixed_50_50_skips_weight_optimization(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=48, freq="h")
    df = pd.DataFrame({"p_norm": np.clip(np.sin(np.linspace(0, 3.14, len(idx))), 0, None)}, index=idx)

    called = {"optimal": False}
    seen_k: list[int] = []

    def fail_optimal(*args, **kwargs):
        called["optimal"] = True
        raise AssertionError("_optimal_weight should not be called in fixed_50_50 mode")

    real_eval = __import__("pv_profiler.block_orientation_fit", fromlist=["_evaluate_candidate_samples"])._evaluate_candidate_samples

    def spy_eval(observed, pred, k):
        seen_k.append(int(k))
        return real_eval(observed, pred, k)

    monkeypatch.setattr("pv_profiler.block_orientation_fit._optimal_weight", fail_optimal)
    monkeypatch.setattr("pv_profiler.block_orientation_fit._evaluate_candidate_samples", spy_eval)

    result, _top, _prof, _single, two_full = run_block5_orientation_fit(
        df,
        latitude=52.45544,
        longitude=13.52481,
        timezone="Etc/GMT-1",
        tilt_step=60,
        az_step=240,
        two_plane_weight_mode="fixed_50_50",
    )

    assert called["optimal"] is False
    assert result["two_plane_weight_mode"] == "fixed_50_50"
    assert 3 not in seen_k
    if not two_full.empty:
        assert set(two_full["weight_opt"]) == {0.5}
        assert set(two_full["weight_mode"]) == {"fixed_50_50"}


def test_two_plane_analytic_optimum_uses_weight_optimization(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=48, freq="h")
    df = pd.DataFrame({"p_norm": np.clip(np.sin(np.linspace(0, 3.14, len(idx))), 0, None)}, index=idx)

    calls = {"optimal": 0}
    seen_k: list[int] = []

    def fake_optimal(*args, **kwargs):
        calls["optimal"] += 1
        return 0.3

    real_eval = __import__("pv_profiler.block_orientation_fit", fromlist=["_evaluate_candidate_samples"])._evaluate_candidate_samples

    def spy_eval(observed, pred, k):
        seen_k.append(int(k))
        return real_eval(observed, pred, k)

    monkeypatch.setattr("pv_profiler.block_orientation_fit._optimal_weight", fake_optimal)
    monkeypatch.setattr("pv_profiler.block_orientation_fit._evaluate_candidate_samples", spy_eval)

    result, _top, _prof, _single, two_full = run_block5_orientation_fit(
        df,
        latitude=52.45544,
        longitude=13.52481,
        timezone="Etc/GMT-1",
        tilt_step=60,
        az_step=240,
        two_plane_weight_mode="analytic_optimum",
    )

    assert calls["optimal"] > 0
    assert result["two_plane_weight_mode"] == "analytic_optimum"
    assert 3 in seen_k
    if not two_full.empty:
        assert set(two_full["weight_mode"]) == {"analytic_optimum"}
