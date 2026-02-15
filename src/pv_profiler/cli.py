"""Command-line interface for pv_profiler."""

from __future__ import annotations

import argparse
import logging

from pv_profiler.config import load_config
from pv_profiler.utils import setup_logging

LOGGER = logging.getLogger(__name__)


def _init(config_path: str) -> dict:
    config = load_config(config_path)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    return config


def cmd_run_single(args: argparse.Namespace) -> int:
    config = _init(args.config)
    from pv_profiler.batch import run_single

    result = run_single(
        system_id=args.system_id,
        input_path=args.input,
        config=config,
        lat=args.lat,
        lon=args.lon,
    )
    LOGGER.info("run-single completed for %s", result["system_id"])
    return 0


def cmd_run_wide(args: argparse.Namespace) -> int:
    config = _init(args.config)
    ids = args.system_ids.split(",") if args.system_ids else None
    from pv_profiler.batch import run_wide

    results = run_wide(input_path=args.input, config=config, system_ids=ids)
    LOGGER.info("run-wide completed for %d systems", len(results))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = _init(args.config)
    from pv_profiler.batch import run_manifest

    results = run_manifest(manifest_path=args.manifest, config=config)
    LOGGER.info("run completed for %d systems", len(results))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    config = _init(args.config)
    output_root = args.output_root or config["paths"]["output_root"]
    from pv_profiler.reporting import write_aggregated_report

    csv_path, json_path = write_aggregated_report(output_root=output_root)
    LOGGER.info("report written: %s and %s", csv_path, json_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pv-ident", description="PV identification pipeline (A-C)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_single_p = subparsers.add_parser("run-single", help="Process one system from long-format file")
    run_single_p.add_argument("--config", "-c", required=True)
    run_single_p.add_argument("--system-id", required=True)
    run_single_p.add_argument("--input", required=True)
    run_single_p.add_argument("--lat", type=float)
    run_single_p.add_argument("--lon", type=float)
    run_single_p.set_defaults(func=cmd_run_single)

    run_wide_p = subparsers.add_parser("run-wide", help="Process all systems from wide-format file")
    run_wide_p.add_argument("--config", "-c", required=True)
    run_wide_p.add_argument("--input", required=True)
    run_wide_p.add_argument("--system-ids", help="Comma-separated subset of system IDs")
    run_wide_p.set_defaults(func=cmd_run_wide)

    run_p = subparsers.add_parser("run", help="Batch processing with manifest.csv")
    run_p.add_argument("--config", "-c", required=True)
    run_p.add_argument("--manifest", required=True)
    run_p.set_defaults(func=cmd_run)

    report_p = subparsers.add_parser("report", help="Aggregate per-system summaries")
    report_p.add_argument("--config", "-c", required=True)
    report_p.add_argument("--output-root")
    report_p.set_defaults(func=cmd_report)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
