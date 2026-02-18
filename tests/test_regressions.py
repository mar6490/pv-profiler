from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pv_profiler.block_io import read_power_timeseries
from pv_profiler.block_orientation import estimate_orientation
from pv_profiler.cli import main


def test_read_power_timeseries_uses_low_memory_false(monkeypatch, tmp_path):
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("timestamp,P_AC\n2015-01-01 00:00:00,1\n", encoding="utf-8")

    captured: dict[str, object] = {}
    real_read_csv = pd.read_csv

    def fake_read_csv(*args, **kwargs):
        captured.update(kwargs)
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr("pv_profiler.block_io.pd.read_csv", fake_read_csv)

    out = read_power_timeseries(csv_path)

    assert out.data.shape[0] == 1
    assert captured.get("low_memory") is False


def test_estimate_orientation_passes_dni_extra(monkeypatch):
    idx = pd.date_range("2020-06-01", periods=4, freq="h", tz="Europe/Berlin")
    power = pd.Series([0.0, 0.4, 0.8, 0.2], index=idx)

    calls: list[dict[str, object]] = []

    def fake_get_total_irradiance(**kwargs):
        calls.append(kwargs)
        assert kwargs.get("dni_extra") is not None
        return {"poa_global": pd.Series([0.1, 0.3, 0.9, 0.2], index=idx)}

    monkeypatch.setattr("pv_profiler.block_orientation.pvlib.irradiance.get_total_irradiance", fake_get_total_irradiance)

    result = estimate_orientation(power, latitude=52.45, longitude=13.52)

    assert result.score >= -1.0
    assert len(calls) > 0


def test_cli_creates_output_directory(monkeypatch, tmp_path):
    output_json = tmp_path / "nested" / "out" / "result.json"

    class DummyResult:
        def to_dict(self):
            return {"ok": True}

    monkeypatch.setattr("pv_profiler.cli.run_single", lambda **_: DummyResult())
    monkeypatch.setattr(
        "sys.argv",
        [
            "pv-ident",
            "run-single",
            "--input-csv",
            "in.csv",
            "--metadata",
            "meta.json",
            "--output-json",
            str(output_json),
        ],
    )

    rc = main()

    assert rc == 0
    assert output_json.exists()
    assert json.loads(output_json.read_text(encoding="utf-8")) == {"ok": True}
