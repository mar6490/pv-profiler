"""Reporting helpers for aggregated summaries."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def collect_summaries(output_root: str | Path) -> pd.DataFrame:
    """Collect all per-system summary.json files under output_root."""
    root = Path(output_root)
    records: list[dict] = []
    for summary_file in root.glob("*/**/summary.json"):
        records.append(json.loads(summary_file.read_text(encoding="utf-8")))
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def write_aggregated_report(output_root: str | Path, report_name: str = "report_summary") -> tuple[Path, Path]:
    """Write aggregated summary report to CSV and JSON."""
    root = Path(output_root)
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    df = collect_summaries(root)

    csv_path = report_dir / f"{report_name}.csv"
    json_path = report_dir / f"{report_name}.json"

    if df.empty:
        df.to_csv(csv_path, index=False)
        json_path.write_text("[]", encoding="utf-8")
    else:
        df.to_csv(csv_path, index=False)
        json_path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")

    return csv_path, json_path
