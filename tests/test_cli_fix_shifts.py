from __future__ import annotations

from pv_profiler.cli import main


class _FakeSdtResult:
    status = "success"
    solver = "CLARABEL"
    fix_shifts = False
    report = {}
    daily_flags = None
    raw_data_matrix = None
    filled_data_matrix = None
    error = None


def test_cli_run_block2_fix_shifts_default_off(monkeypatch, tmp_path):
    input_parquet = tmp_path / "01_input_power.parquet"
    input_parquet.write_bytes(b"placeholder")

    seen: dict[str, object] = {}

    def fake_run_block2_sdt_from_parquet(**kwargs):
        seen.update(kwargs)
        r = _FakeSdtResult()
        r.fix_shifts = bool(kwargs.get("fix_shifts"))
        return r

    monkeypatch.setattr("pv_profiler.cli.run_block2_sdt_from_parquet", fake_run_block2_sdt_from_parquet)
    monkeypatch.setattr(
        "sys.argv",
        [
            "pv-ident",
            "run-block2",
            "--input-parquet",
            str(input_parquet),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    rc = main()

    assert rc == 0
    assert seen["fix_shifts"] is False


def test_cli_run_block2_fix_shifts_opt_in(monkeypatch, tmp_path):
    input_parquet = tmp_path / "01_input_power.parquet"
    input_parquet.write_bytes(b"placeholder")

    seen: dict[str, object] = {}

    def fake_run_block2_sdt_from_parquet(**kwargs):
        seen.update(kwargs)
        r = _FakeSdtResult()
        r.fix_shifts = bool(kwargs.get("fix_shifts"))
        return r

    monkeypatch.setattr("pv_profiler.cli.run_block2_sdt_from_parquet", fake_run_block2_sdt_from_parquet)
    monkeypatch.setattr(
        "sys.argv",
        [
            "pv-ident",
            "run-block2",
            "--input-parquet",
            str(input_parquet),
            "--output-dir",
            str(tmp_path / "out"),
            "--fix-shifts",
        ],
    )

    rc = main()

    assert rc == 0
    assert seen["fix_shifts"] is True


def test_cli_run_batch_fix_shifts_default_off(monkeypatch, tmp_path):
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
    assert seen["fix_shifts"] is False


def test_cli_run_batch_fix_shifts_opt_in(monkeypatch, tmp_path):
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
            "--fix-shifts",
        ],
    )

    rc = main()

    assert rc == 0
    assert seen["fix_shifts"] is True
