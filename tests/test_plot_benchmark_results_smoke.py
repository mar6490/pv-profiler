from __future__ import annotations

import subprocess
import sys

import pandas as pd


def test_plot_benchmark_results_ew_smoke(tmp_path):
    df = pd.DataFrame(
        [
            {
                "system_id": 1,
                "model_type": "two_plane",
                "center_true": 180.0,
                "center_est": 182.0,
                "az_center_abs_err_deg": 2.0,
                "az_plane_abs_err_deg": 3.0,
                "weight_true": 0.5,
                "weight_est": 0.45,
                "weight_abs_err": 0.05,
                "tilt_true": 25.0,
                "tilt_abs_err_deg": 1.0,
            },
            {
                "system_id": 2,
                "model_type": "two_plane",
                "center_true": 10.0,
                "center_est": 188.0,
                "az_center_abs_err_deg": 178.0,
                "az_plane_abs_err_deg": 8.0,
                "weight_true": 0.6,
                "weight_est": 0.55,
                "weight_abs_err": 0.05,
                "tilt_true": 20.0,
                "tilt_abs_err_deg": 2.0,
            },
        ]
    )
    input_csv = tmp_path / "benchmark_results.csv"
    df.to_csv(input_csv, index=False)

    subprocess.run(
        [sys.executable, "scripts/plot_benchmark_results.py", "--input", str(input_csv)],
        check=True,
    )

    plots = tmp_path / "plots"
    assert plots.exists()
    assert (plots / "hist_plane_err.png").exists()
