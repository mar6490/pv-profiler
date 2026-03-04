from __future__ import annotations

import json

import pandas as pd
import pytest

from pv_profiler.diagnostics_v2 import generate_diagnostics_v2


def _prepare_fixture(tmp_path):
    out = tmp_path / "batch_out"
    s1 = out / "system_001"
    s2 = out / "system_002"
    s1.mkdir(parents=True)
    s2.mkdir(parents=True)

    (s1 / "00_status.json").write_text(
        json.dumps({"status": "ok", "runtime_seconds": 1.2, "error": None}),
        encoding="utf-8",
    )
    (s1 / "08_orientation_result.json").write_text(
        json.dumps({"model_type": "single", "tilt_deg": 20.0, "azimuth_deg": 180.0, "score_rmse": 0.12}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "tilt_deg": [10, 10, 20, 20],
            "azimuth_deg": [160, 180, 160, 180],
            "rmse": [0.20, 0.15, 0.18, 0.16],
            "bic": [12.0, 8.0, 11.0, 9.0],
        }
    ).to_csv(s1 / "09a_orientation_single_full_grid.csv", index=False)
    pd.DataFrame(
        {
            "tilt_deg": [10, 10, 20, 20],
            "azimuth_center_deg": [80, 100, 80, 100],
            "weight_mode": ["fixed_50_50"] * 4,
            "weight_opt": [0.5] * 4,
            "rmse": [0.14, 0.17, 0.13, 0.18],
            "bic": [7.0, 10.0, 6.0, 11.0],
        }
    ).to_csv(s1 / "09b_orientation_two_plane_full_grid.csv", index=False)

    (s2 / "00_status.json").write_text(
        json.dumps({"status": "failed", "runtime_seconds": 0.3, "error": {"type": "ValueError", "message": "x"}}),
        encoding="utf-8",
    )

    meta = tmp_path / "systems_metadata.csv"
    pd.DataFrame(
        {
            "system_id": [1, 2],
            "site_name": ["A", "B"],
            "tilt_true": [20, 30],
            "center_true": [90, 85],
            "azimuth_true": [180, 170],
        }
    ).to_csv(meta, index=False)
    return out, meta


def test_generate_diagnostics_v2_core_plots_default_and_health_opt_in(tmp_path):
    out, meta = _prepare_fixture(tmp_path)

    df = generate_diagnostics_v2(output_root=out, systems_metadata_csv=meta, system_id_col="system_id")

    assert len(df) == 2
    assert (out / "diagnostics_v2" / "aggregated_metrics.csv").exists()

    assert {
        "best_single_rmse",
        "best_two_rmse",
        "delta_rmse",
        "delta_bic",
        "rmse_prefers_two_plane",
        "bic_prefers_two_plane",
    }.issubset(set(df.columns))

    row1 = df[df["system_dir"] == "system_001"].iloc[0]
    assert row1["best_single_rmse"] == 0.15
    assert row1["best_two_rmse"] == 0.13
    assert row1["delta_rmse"] == pytest.approx(-0.02)
    assert bool(row1["rmse_prefers_two_plane"])

    # core per-system plots (always on)
    assert (out / "diagnostics_v2" / "per_system" / "system_001" / "rmse_single_landscape.png").exists()
    assert (out / "diagnostics_v2" / "per_system" / "system_001" / "bic_single_landscape.png").exists()
    assert (out / "diagnostics_v2" / "per_system" / "system_001" / "rmse_two_plane_landscape.png").exists()
    assert (out / "diagnostics_v2" / "per_system" / "system_001" / "bic_two_plane_landscape.png").exists()
    assert (out / "diagnostics_v2" / "per_system" / "system_001" / "single_1d_azimuth_rmse_bic.png").exists()
    assert (out / "diagnostics_v2" / "per_system" / "system_001" / "two_plane_1d_center_rmse_bic.png").exists()

    # missing grids: row kept, plots skipped
    row2 = df[df["system_dir"] == "system_002"].iloc[0]
    assert pd.isna(row2["delta_rmse"])
    assert not (out / "diagnostics_v2" / "per_system" / "system_002" / "rmse_single_landscape.png").exists()

    # core global plots (always on)
    assert (out / "diagnostics_v2" / "global" / "scatter_delta_rmse_vs_delta_bic.png").exists()
    assert (out / "diagnostics_v2" / "global" / "hist_delta_rmse.png").exists()
    assert (out / "diagnostics_v2" / "global" / "hist_delta_bic.png").exists()
    assert (out / "diagnostics_v2" / "global" / "model_choice_map.png").exists()

    # health plots default off
    assert not (out / "diagnostics_v2" / "global" / "runtime_hist.png").exists()
    assert not (out / "diagnostics_v2" / "global" / "status_counts.png").exists()
    assert not (out / "diagnostics_v2" / "per_system" / "system_001" / "artifact_presence.png").exists()
    assert not (out / "diagnostics_v2" / "per_system" / "system_001" / "daily_flags.png").exists()

