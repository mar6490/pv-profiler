from __future__ import annotations

from pathlib import Path

from pv_profiler.pipeline import run_single
from pv_profiler.types import OrientationResult


DATA_CSV = Path("data/sonnja_pv3_2015/einleuchtend_wrdata_2015_wr1_5min_naive.csv")
META_JSON = Path("data/sonnja_pv3_2015/metadata.json")


def test_smoke_run_single(monkeypatch):
    def fake_sdt(series):
        return series.iloc[:288], []

    def fake_orientation(power_series, latitude, longitude, altitude=None):
        assert latitude != 0
        assert longitude != 0
        assert len(power_series) > 0
        return OrientationResult(tilt=15.0, azimuth=215.0, score=0.8)

    monkeypatch.setattr("pv_profiler.pipeline.run_sdt_onboarding", fake_sdt)
    monkeypatch.setattr("pv_profiler.pipeline.estimate_orientation", fake_orientation)

    result = run_single(DATA_CSV, META_JSON)

    assert result.orientation.tilt == 15.0
    assert result.orientation.azimuth == 215.0
    assert result.diagnostics.n_samples == 288
