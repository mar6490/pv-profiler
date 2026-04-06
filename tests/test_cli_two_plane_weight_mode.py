from __future__ import annotations

from pv_profiler.cli import main


def test_cli_run_block5_no_removed_two_plane_args(monkeypatch, tmp_path):
    seen: dict[str, object] = {}

    def fake_run_block5_from_files(**kwargs):
        seen.update(kwargs)
        return {"model_type": "single", "timing_seconds": {"total": 0.0}}

    monkeypatch.setattr("pv_profiler.cli.run_block5_from_files", fake_run_block5_from_files)
    monkeypatch.setattr(
        "sys.argv",
        [
            "pv-ident",
            "run-block5",
            "--input-p-norm-parquet",
            str(tmp_path / "07_p_norm_clear.parquet"),
            "--output-dir",
            str(tmp_path / "out"),
            "--latitude",
            "52.45544",
            "--longitude",
            "13.52481",
        ],
    )

    rc = main()

    assert rc == 0
    assert "two_plane_weight_mode" not in seen
    assert "two_plane_delta_az_deg" not in seen


def test_cli_run_batch_no_removed_two_plane_args(monkeypatch, tmp_path):
    seen: dict[str, object] = {}

    def fake_run_batch(**kwargs):
        import pandas as pd

        seen.update(kwargs)
        return pd.DataFrame([{"system_id": 1, "status": "ok", "runtime_seconds": 0.1}])

    monkeypatch.setattr("pv_profiler.cli.run_batch", fake_run_batch)
    monkeypatch.setattr(
        "sys.argv",
        [
            "pv-ident",
            "run-batch",
            "--input-dir",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "out"),
        ],
    )

    rc = main()

    assert rc == 0
    assert "two_plane_weight_mode" not in seen
