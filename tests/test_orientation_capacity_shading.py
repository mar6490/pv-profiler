from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from pv_profiler.capacity import estimate_kWp_effective
from pv_profiler.orientation import fit_orientation
from pv_profiler.shading import compute_shading


class FakeLocation:
    def __init__(self, latitude: float, longitude: float, tz: str) -> None:
        self.latitude = latitude
        self.longitude = longitude
        self.tz = tz

    def get_solarposition(self, times: pd.DatetimeIndex) -> pd.DataFrame:
        n = len(times)
        return pd.DataFrame(
            {
                "apparent_zenith": np.linspace(70, 10, n),
                "apparent_elevation": np.linspace(10, 70, n),
                "azimuth": np.linspace(95, 265, n),
            },
            index=times,
        )

    def get_clearsky(self, times: pd.DatetimeIndex, model: str = "ineichen") -> pd.DataFrame:
        _ = model
        n = len(times)
        return pd.DataFrame(
            {
                "dni": np.full(n, 800.0),
                "ghi": np.full(n, 700.0),
                "dhi": np.full(n, 120.0),
            },
            index=times,
        )


class FakeIrradiance:
    @staticmethod
    def get_total_irradiance(
        surface_tilt,
        surface_azimuth,
        solar_zenith,
        solar_azimuth,
        dni,
        ghi,
        dhi,
        model,
    ):
        _ = (solar_zenith, solar_azimuth, dni, ghi, dhi, model)
        poa = 1000.0 - 2.0 * abs(surface_azimuth - 180.0) - 3.0 * abs(surface_tilt - 30.0)
        values = np.full(len(solar_zenith), max(50.0, poa))
        return {"poa_global": pd.Series(values, index=solar_zenith.index)}


class FakeFigure:
    def colorbar(self, *args, **kwargs):
        _ = (args, kwargs)

    def tight_layout(self):
        return None

    def savefig(self, path, dpi=150):
        Path(path).write_text("fake_png", encoding="utf-8")


class FakeAxes:
    def imshow(self, *args, **kwargs):
        _ = (args, kwargs)
        return object()

    def set_xlabel(self, *args, **kwargs):
        _ = (args, kwargs)

    def set_ylabel(self, *args, **kwargs):
        _ = (args, kwargs)

    def set_title(self, *args, **kwargs):
        _ = (args, kwargs)


class FakePlt:
    @staticmethod
    def subplots(figsize=(9, 4.5)):
        _ = figsize
        return FakeFigure(), FakeAxes()

    @staticmethod
    def close(fig):
        _ = fig


def test_orientation_grid_search_smoke(monkeypatch) -> None:
    fake_pvlib = SimpleNamespace(location=SimpleNamespace(Location=FakeLocation), irradiance=FakeIrradiance())
    monkeypatch.setitem(__import__("sys").modules, "pvlib", fake_pvlib)

    times = pd.date_range("2024-06-01 08:00:00", periods=24, freq="5min", tz="Etc/GMT-1")
    ac = pd.Series(np.linspace(100, 450, len(times)), index=times)
    fit_mask = pd.Series(True, index=times)

    result = fit_orientation(ac, fit_mask, lat=52.5, lon=13.4, config={"orientation": {"top_n": 5}})
    assert result.result["model_type"] in {"single", "two-plane"}
    assert 0 <= result.result["tilt_deg"] <= 60
    assert "score_rmse" in result.result
    assert len(result.diagnostics) <= 5


def test_capacity_estimation_smoke() -> None:
    times = pd.date_range("2024-06-01 10:00:00", periods=12, freq="5min", tz="Etc/GMT-1")
    poa = pd.Series(np.full(len(times), 900.0), index=times)
    expected_kwp = 4.2
    ac = pd.Series((poa / 1000.0) * expected_kwp, index=times)
    fit_mask = pd.Series(True, index=times)

    result = estimate_kWp_effective(ac, poa, fit_mask, poa_threshold_wm2=600.0)
    assert abs(result["kWp_effective"] - expected_kwp) < 1e-6
    assert result["kWp_effective_n_points"] == len(times)


def test_shading_map_binning(monkeypatch, tmp_path: Path) -> None:
    fake_pvlib = SimpleNamespace(location=SimpleNamespace(Location=FakeLocation), irradiance=FakeIrradiance())
    monkeypatch.setitem(__import__("sys").modules, "pvlib", fake_pvlib)
    monkeypatch.setitem(__import__("sys").modules, "matplotlib.pyplot", FakePlt)

    times = pd.date_range("2024-06-01 08:00:00", periods=12, freq="5min", tz="Etc/GMT-1")
    poa = pd.Series(np.full(len(times), 800.0), index=times)
    ac = pd.Series(np.full(len(times), 3.6), index=times)
    clear = pd.Series(True, index=times)

    shading_map, metrics = compute_shading(
        ac_power_clean=ac,
        fit_times=clear,
        poa_cs=poa,
        kWp_effective=4.0,
        lat=52.5,
        lon=13.4,
        config={"shading": {"az_bin_deg": 10, "el_bin_deg": 10, "r_max": 1.2}},
        plot_path=tmp_path / "shading_map.png",
    )

    assert not shading_map.empty
    assert int(shading_map["count"].sum()) == len(times)
    assert "global_shading_index" in metrics
    assert (tmp_path / "shading_map.png").exists()
