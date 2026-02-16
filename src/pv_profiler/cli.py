from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run_single


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
            Path(args.output_json).write_text(output + "\n", encoding="utf-8")
        else:
            print(output)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
