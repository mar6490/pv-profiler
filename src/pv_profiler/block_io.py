from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .types import InputDiagnostics, InputLoaderResult, TimeSeriesData


def _detect_csv_options(path: Path) -> tuple[str, str, str, str]:
    encodings = ("utf-8", "latin-1")
    last_exc: Exception | None = None
    for enc in encodings:
        try:
            sample = path.read_text(encoding=enc)[:4096]
            break
        except Exception as exc:  # pragma: no cover - defensive
            last_exc = exc
    else:  # pragma: no cover - defensive
        raise ValueError(f"Could not read CSV header for dialect detection: {last_exc}")

    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(sample, delimiters=",;\t|")
        delimiter = dialect.delimiter
        quotechar = dialect.quotechar or '"'
    except csv.Error:
        delimiter = ","
        quotechar = '"'

    decimal = "," if delimiter == ";" else "."
    return delimiter, enc, decimal, quotechar


def load_input_for_sdt(
    input_path: str | Path,
    timestamp_col: str = "timestamp",
    power_col: str = "P_AC",
    timezone: str | None = None,
    resample_if_irregular: bool = True,
    min_samples: int = 288,
    clip_negative_power: bool = True,
) -> InputLoaderResult:
    path = Path(input_path)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Unsupported file type for MVP: {path.suffix}. Only CSV is supported.")

    delimiter, encoding, decimal, quotechar = _detect_csv_options(path)
    df = pd.read_csv(
        path,
        usecols=[timestamp_col, power_col],
        sep=delimiter,
        encoding=encoding,
        decimal=decimal,
        quotechar=quotechar,
        low_memory=False,
    )

    decisions: list[str] = [
        f"csv_options: sep='{delimiter}', encoding='{encoding}', decimal='{decimal}', quotechar='{quotechar}'",
        f"selected_columns: [{timestamp_col}, {power_col}]",
        "output_column_standardized_to: power",
    ]

    ts = pd.to_datetime(df[timestamp_col], errors="coerce")
    power = pd.to_numeric(df[power_col], errors="coerce")
    out = pd.DataFrame({"power": power.to_numpy()}, index=ts)

    invalid_ts = int(out.index.isna().sum())
    if invalid_ts:
        decisions.append(f"invalid_timestamps_dropped: {invalid_ts}")
    out = out[~out.index.isna()].sort_index()

    if timezone is not None:
        if out.index.tz is None:
            out.index = out.index.tz_localize(timezone, ambiguous="NaT", nonexistent="shift_forward")
            decisions.append(f"tz_localized_to: {timezone}")
        else:
            out.index = out.index.tz_convert(timezone)
            decisions.append(f"tz_converted_to: {timezone}")

    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
        decisions.append("tz_removed_for_sdt: true (using naive local-time index)")

    if clip_negative_power:
        n_neg = int((out["power"] < 0).sum(skipna=True))
        if n_neg:
            out["power"] = out["power"].clip(lower=0)
        decisions.append(f"negatives_clipped: {bool(n_neg)} (count={n_neg})")

    diffs = out.index.to_series().diff().dropna()
    sampling_counts = diffs.value_counts().head(10)
    sampling_summary = {str(k): int(v) for k, v in sampling_counts.items()}
    dominant_delta = sampling_counts.index[0] if not sampling_counts.empty else None
    dominant_share = float(sampling_counts.iloc[0] / len(diffs)) if len(diffs) > 0 else 0.0

    if dominant_delta is not None and dominant_share < 0.95:
        if resample_if_irregular:
            out = out.resample(dominant_delta).mean()
            decisions.append(
                f"resampled_to_dominant_delta: {dominant_delta} (dominant_share_before={dominant_share:.2%})"
            )
        else:
            raise ValueError(
                "Irregular sampling detected. "
                f"Dominant delta {dominant_delta} only covers {dominant_share:.2%}; "
                "set resample_if_irregular=True to regularize."
            )

    if len(out) < min_samples:
        raise ValueError(
            f"Not enough samples for SDT onboarding: len={len(out)} < min_samples={min_samples}."
        )

    share_non_null = float(out["power"].notna().mean()) if len(out) else 0.0
    if share_non_null < 0.5:
        raise ValueError(f"Too many NaNs in power series: non-null share={share_non_null:.2%}")

    decisions.append("power_nans_kept: true (left for downstream SDT handling)")

    diagnostics = InputDiagnostics(
        shape=out.shape,
        columns=list(out.columns),
        min_time=out.index.min().isoformat() if len(out) else None,
        max_time=out.index.max().isoformat() if len(out) else None,
        dominant_timedelta=str(dominant_delta) if dominant_delta is not None else None,
        sampling_summary=sampling_summary,
        share_positive_power=float((out["power"] > 0).mean()) if len(out) else 0.0,
        share_nan_power=float(out["power"].isna().mean()) if len(out) else 0.0,
        share_non_null_power=share_non_null,
        index_monotonic_increasing=bool(out.index.is_monotonic_increasing),
        min_power=float(out["power"].min()) if out["power"].notna().any() else None,
        max_power=float(out["power"].max()) if out["power"].notna().any() else None,
        decisions=decisions,
    )
    return InputLoaderResult(data=out, diagnostics=diagnostics)


def write_input_loader_artifacts(result: InputLoaderResult, output_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / "01_input_power.parquet"
    diagnostics_path = out_dir / "01_input_diagnostics.json"

    result.data.to_parquet(parquet_path, index=True)
    diagnostics_payload = {
        "shape": list(result.diagnostics.shape),
        "columns": result.diagnostics.columns,
        "min_time": result.diagnostics.min_time,
        "max_time": result.diagnostics.max_time,
        "dominant_timedelta": result.diagnostics.dominant_timedelta,
        "sampling_summary": result.diagnostics.sampling_summary,
        "share_positive_power": result.diagnostics.share_positive_power,
        "share_nan_power": result.diagnostics.share_nan_power,
        "share_non_null_power": result.diagnostics.share_non_null_power,
        "index_monotonic_increasing": result.diagnostics.index_monotonic_increasing,
        "min_power": result.diagnostics.min_power,
        "max_power": result.diagnostics.max_power,
        "decisions": result.diagnostics.decisions,
    }
    diagnostics_path.write_text(json.dumps(diagnostics_payload, indent=2) + "\n", encoding="utf-8")

    return parquet_path, diagnostics_path


def read_power_timeseries(csv_path: str | Path, power_column: str = "P_AC") -> TimeSeriesData:
    loaded = load_input_for_sdt(
        input_path=csv_path,
        timestamp_col="timestamp",
        power_col=power_column,
        timezone=None,
        resample_if_irregular=True,
        min_samples=1,
        clip_negative_power=True,
    )
    return TimeSeriesData(data=loaded.data["power"], timezone=None, source_column=power_column)


def read_metadata(metadata_path: str | Path | None) -> dict[str, Any]:
    if metadata_path is None:
        return {}
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
