from __future__ import annotations

from pathlib import Path

import pandas as pd

from pv_profiler import batch


def _fake_write_parquet(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = df.to_frame() if hasattr(df, "to_frame") and not hasattr(df, "columns") else df
    frame.to_csv(path, index=True)


def _fake_block_a(power: pd.Series, lat: float, lon: float, out_dir: Path, config=None):
    _ = (lat, lon, out_dir, config)
    idx = power.index
    clipped = pd.Series(False, index=idx, name="is_clipped_time")
    fit_times = pd.Series(True, index=idx, name="is_fit_time")
    return {
        "parsed": power.rename("ac_power").to_frame(),
        "ac_power_clean": power.rename("ac_power_clean").to_frame(),
        "fit_times": fit_times.to_frame(),
        "clipped_times": clipped.to_frame(),
        "daily_flags": pd.DataFrame({"date": [str(idx[0].date())], "clipping_day_share": [0.0]}),
        "clipping_summary": {"clipping_day_share_mean": 0.0, "clipping_fraction_day_median": 0.0, "n_days": 1},
        "sdt_summary": {
            "n_days_total": 1,
            "n_clear_days": 1,
            "clear_day_fraction": 1.0,
            "n_fit_times": int(len(idx)),
            "clear_days_source": "sdt:daily_flags:clear_day",
            "time_shift_correction_applied": False,
            "time_zone_correction": 0.0,
        },
        "sdt_introspect": {"clear_days_source": "sdt:daily_flags:clear_day"},
    }


def test_run_single_smoke(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(batch, "write_parquet", _fake_write_parquet)
    monkeypatch.setattr(batch, "run_block_a", _fake_block_a)
    monkeypatch.setattr(batch, "compute_daily_peak_and_norm", lambda **kwargs: (pd.DataFrame({"metric":["x"],"value":[1.0]}), pd.Series([1.0,1.0,1.0], index=kwargs["ac_power_clean"].index, name="p_norm")))
    monkeypatch.setattr(batch, "compute_fit_mask", lambda **kwargs: (pd.Series([True, True, True], index=kwargs["p_norm"].index, name="fit_mask"), pd.DataFrame({"date":[str(kwargs["p_norm"].index[0].date())],"daily_fit_fraction":[1.0]})))
    monkeypatch.setattr(batch, "fit_orientation", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("stop-after-c")))

    input_csv = tmp_path / "example_single.csv"
    pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01T12:00:00",
                "2024-01-01T12:05:00",
                "2024-01-01T12:10:00",
            ],
            "P_AC": [100.0, 120.0, 110.0],
        }
    ).to_csv(input_csv, index=False)

    plants_csv = tmp_path / "plants.csv"
    pd.DataFrame(
        {
            "system_id": ["sys1"],
            "country": ["DE"],
            "plz": ["12345"],
            "lat": [52.5],
            "lon": [13.4],
            "timezone": ["Europe/Berlin"],
        }
    ).to_csv(plants_csv, index=False)

    cfg = {
        "paths": {"output_root": str(tmp_path / "outputs"), "plants_csv": str(plants_csv)},
        "pipeline": {"fit_tau": 0.03},
    }

    try:
        batch.run_single(system_id="sys1", input_path=str(input_csv), config=cfg, lat=None, lon=None)
    except RuntimeError as exc:
        assert str(exc) == "stop-after-c"

    run_dir = next((tmp_path / "outputs" / "sys1").glob("*"))
    expected = [
        "01_parsed_tzaware.parquet",
        "02_cleaned_timeshift_fixed.parquet",
        "03_fit_times_mask.parquet",
        "04_daily_flags.csv",
        "05_clipping_summary.json",
        "06_clipped_times_mask.parquet",
        "07_sdt_summary.json",
        "07_sdt_introspect.json",
        "08_daily_peak.csv",
        "09_p_norm.parquet",
        "11_fit_mask.parquet",
        "12_daily_fit_fraction.csv",
    ]
    for filename in expected:
        assert (run_dir / filename).exists(), filename
