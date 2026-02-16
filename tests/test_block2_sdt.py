from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pv_profiler.block_sdt import run_block2_sdt, write_block2_artifacts
from pv_profiler.cli import main


class _FakeDataHandler:
    def __init__(self, data_frame):
        self.data_frame = data_frame
        self.raw_data_matrix = [[1.0, 2.0], [3.0, 4.0]]
        self.filled_data_matrix = [[1.1, 2.1], [3.1, 4.1]]
        class DailyFlags:
            def __init__(self):
                self.clear = [True, False]
                self.cloudy = [False, True]
                self.no_errors = [True, True]
                self.density = [False, True]

        self.daily_flags = DailyFlags()

    def run_pipeline(self, power_col: str, fix_shifts: bool, solver: str):
        assert power_col == "power"
        assert solver == "CLARABEL"

    def report(self, return_values: bool = True, verbose: bool = False):
        assert return_values is True
        assert verbose is False
        return {"capacity_estimate": 4.2, "data_quality_score": 0.9}


class _MemoryErrorDataHandler(_FakeDataHandler):
    def run_pipeline(self, power_col: str, fix_shifts: bool, solver: str):
        raise MemoryError("oom in clipping")


def _sample_power_df() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=24, freq="h")
    return pd.DataFrame({"power": [float(i % 6) for i in range(len(idx))]}, index=idx)


def test_run_block2_sdt_writes_report_and_daily_flags(monkeypatch, tmp_path):
    monkeypatch.setattr("pv_profiler.block_sdt.DataHandler", _FakeDataHandler, raising=False)
    # patch import target inside function
    monkeypatch.setitem(__import__("sys").modules, "solardatatools", type("M", (), {"DataHandler": _FakeDataHandler})())

    result = run_block2_sdt(_sample_power_df(), solver="CLARABEL", fix_shifts=True, power_col="power")
    written = write_block2_artifacts(result, tmp_path)

    assert result.status == "success"
    assert Path(written["report"]).exists()
    assert Path(written["daily_flags"]).exists()
    assert Path(written["raw_data_matrix"]).exists()
    assert Path(written["filled_data_matrix"]).exists()

    flags = pd.read_csv(Path(written["daily_flags"]), index_col=0)
    assert {"clear", "cloudy", "no_errors"}.issubset(set(flags.columns))
    assert result.raw_data_matrix is not None
    assert flags.shape[0] == result.raw_data_matrix.shape[1]


def test_run_block2_sdt_memoryerror_writes_error(monkeypatch, tmp_path):
    monkeypatch.setitem(
        __import__("sys").modules,
        "solardatatools",
        type("M", (), {"DataHandler": _MemoryErrorDataHandler})(),
    )

    result = run_block2_sdt(_sample_power_df(), solver="CLARABEL", fix_shifts=True, power_col="power")
    written = write_block2_artifacts(result, tmp_path)

    assert result.status in {"failed", "partial"}
    assert "error" in written
    payload = json.loads(Path(written["error"]).read_text(encoding="utf-8"))
    assert payload["exception_type"] == "MemoryError"


def test_cli_run_block2_parquet_path(monkeypatch, tmp_path):
    input_parquet = tmp_path / "01_input_power.parquet"
    _sample_power_df().to_parquet(input_parquet)

    monkeypatch.setitem(__import__("sys").modules, "solardatatools", type("M", (), {"DataHandler": _FakeDataHandler})())

    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "pv-ident",
            "run-block2",
            "--input-parquet",
            str(input_parquet),
            "--output-dir",
            str(out_dir),
        ],
    )

    rc = main()

    assert rc == 0
    assert (out_dir / "02_sdt_report.json").exists()
    assert (out_dir / "02_sdt_daily_flags.csv").exists() or (out_dir / "02_sdt_error.json").exists()
