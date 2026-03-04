from __future__ import annotations

import json

import pandas as pd

from pv_profiler.diagnostics_v2 import generate_diagnostics_v2


def test_generate_diagnostics_v2_smoke(tmp_path):
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
    pd.DataFrame({"day": ["2015-01-01"], "is_clear": [True]}).to_csv(s1 / "02_sdt_daily_flags.csv", index=False)

    (s2 / "00_status.json").write_text(
        json.dumps({"status": "failed", "runtime_seconds": 0.3, "error": {"type": "ValueError", "message": "x"}}),
        encoding="utf-8",
    )

    meta = tmp_path / "systems_metadata.csv"
    pd.DataFrame({"system_id": [1, 2], "site_name": ["A", "B"]}).to_csv(meta, index=False)

    df = generate_diagnostics_v2(output_root=out, systems_metadata_csv=meta, system_id_col="system_id")

    assert len(df) == 2
    assert (out / "diagnostics_v2" / "aggregated_metrics.csv").exists()
    assert (out / "diagnostics_v2" / "global" / "status_counts.png").exists()
    assert (out / "diagnostics_v2" / "per_system" / "system_001" / "artifact_presence.png").exists()
    assert "site_name" in df.columns
    assert set(df["status"].tolist()) == {"ok", "failed"}
