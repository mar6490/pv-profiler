"""Command-line interface for pv_profiler."""

import argparse
import logging

from pv_profiler.config import load_config
from pv_profiler.utils import setup_logging


LOGGER = logging.getLogger(__name__)


def _run_placeholder(args: argparse.Namespace) -> int:
    load_config(args.config)
    setup_logging()
    LOGGER.info("Not yet implemented")
    print("Not yet implemented")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pv-ident", description="PV profiler CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command_name in ("run", "run-single", "run-wide", "report"):
        command_parser = subparsers.add_parser(command_name)
        command_parser.add_argument("-c", "--config", required=True, help="Path to config YAML")
        command_parser.set_defaults(func=_run_placeholder)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
