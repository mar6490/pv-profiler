from __future__ import annotations

import argparse

from pv_profiler.benchmark import build_benchmark_results


def main() -> int:
    p = argparse.ArgumentParser(description="Build benchmark_results.csv from batch outputs")
    p.add_argument("--output-root", required=True)
    p.add_argument("--systems-metadata-csv", required=True)
    p.add_argument("--system-id-col", default="system_id")
    args = p.parse_args()

    build_benchmark_results(
        output_root=args.output_root,
        systems_metadata_csv=args.systems_metadata_csv,
        system_id_col=args.system_id_col,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
