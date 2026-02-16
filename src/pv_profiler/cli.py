from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run_block1_input_loader, run_single


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pv-ident", description="PV orientation profiler (MVP)")
    sub = parser.add_subparsers(dest="command", required=True)

    run_single_parser = sub.add_parser("run-single", help="Run profiling for one AC power CSV")
    run_single_parser.add_argument("--input-csv", required=True, help="Input CSV with timestamp and AC power column")
    run_single_parser.add_argument("--metadata", required=True, help="Path to metadata JSON with lat/lon")
    run_single_parser.add_argument("--power-column", default="P_AC", help="Power column name in input CSV")
    run_single_parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output path for JSON result (prints to stdout if omitted)",
    )

    block1_parser = sub.add_parser("run-block1", help="Run Block 1 input loader and write artifacts")
    block1_parser.add_argument("--input-csv", required=True, help="Input CSV with timestamp and power columns")
    block1_parser.add_argument("--output-dir", required=True, help="Directory for Block 1 artifacts")
    block1_parser.add_argument("--timestamp-col", default="timestamp", help="Timestamp column in CSV")
    block1_parser.add_argument("--power-col", default="P_AC", help="Power column in CSV")
    block1_parser.add_argument("--timezone", default=None, help="Optional timezone to localize/convert")
    block1_parser.add_argument(
        "--no-resample",
        action="store_true",
        help="Fail on irregular sampling instead of resampling to dominant interval",
    )
    block1_parser.add_argument(
        "--min-samples",
        type=int,
        default=288,
        help="Minimum required number of rows after cleaning",
    )
    block1_parser.add_argument(
        "--keep-negative-power",
        action="store_true",
        help="Keep negative power values instead of clipping to zero",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run-single":
        result = run_single(
            input_csv=args.input_csv,
            metadata_path=args.metadata,
            power_column=args.power_column,
        )
        output = json.dumps(result.to_dict(), indent=2)
        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output + "\n", encoding="utf-8")
        else:
            print(output)
        return 0

    if args.command == "run-block1":
        result = run_block1_input_loader(
            input_csv=args.input_csv,
            output_dir=args.output_dir,
            timestamp_col=args.timestamp_col,
            power_col=args.power_col,
            timezone=args.timezone,
            resample_if_irregular=not args.no_resample,
            min_samples=args.min_samples,
            clip_negative_power=not args.keep_negative_power,
        )
        summary = {
            "shape": list(result.diagnostics.shape),
            "min_time": result.diagnostics.min_time,
            "max_time": result.diagnostics.max_time,
            "dominant_timedelta": result.diagnostics.dominant_timedelta,
            "share_positive_power": result.diagnostics.share_positive_power,
            "share_nan_power": result.diagnostics.share_nan_power,
        }
        print(json.dumps(summary, indent=2))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
