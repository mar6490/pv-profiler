from __future__ import annotations

import pandas as pd
import pvlib

from pv_profiler.block_orientation_fit import run_block5_from_files


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
        az_step=60,
        topk=5,
    )

    assert result["model_type"] == "two_plane"
    assert abs(float(result["tilt_deg"]) - 20) <= 10
    assert abs(float(result["azimuth_center_deg"]) - 180) <= 20
    assert 0.2 <= float(result["weight_east"]) <= 0.8
