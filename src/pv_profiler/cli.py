from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import (
    run_block1_input_loader,
    run_block2_sdt_from_csv,
    run_block2_sdt_from_parquet,
    run_block3_from_files,
    run_block4_from_files,
    run_block5_from_files,
    run_single,
)


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

    block2_parser = sub.add_parser("run-block2", help="Run SDT onboarding (Block 2) and write artifacts")
    source = block2_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-parquet", help="Path to 01_input_power.parquet from Block 1")
    source.add_argument("--input-csv", help="Raw CSV input (will run Block 1 internally before Block 2)")
    block2_parser.add_argument("--output-dir", required=True, help="Directory for Block 2 artifacts")
    block2_parser.add_argument("--solver", default="CLARABEL", help="SDT solver passed to DataHandler.run_pipeline")
    block2_parser.add_argument(
        "--no-fix-shifts",
        action="store_true",
        help="Disable SDT timestamp shift correction",
    )
    block2_parser.add_argument("--power-col", default="power", help="Power column for parquet input")
    block2_parser.add_argument("--timestamp-col", default="timestamp", help="Timestamp column for CSV input")
    block2_parser.add_argument("--csv-power-col", default="P_AC", help="Power column for CSV input")
    block2_parser.add_argument("--timezone", default=None, help="Optional timezone for CSV input")
    block2_parser.add_argument("--min-samples", type=int, default=288, help="Minimum samples for Block 1 CSV path")
    block2_parser.add_argument(
        "--no-resample",
        action="store_true",
        help="Fail on irregular sampling in Block 1 CSV path",
    )
    block2_parser.add_argument(
        "--keep-negative-power",
        action="store_true",
        help="Keep negative power values in Block 1 CSV path",
    )

    block3_parser = sub.add_parser("run-block3", help="Run fit-day selection (Block 3)")
    block3_parser.add_argument("--input-power-parquet", required=True, help="Path to 01_input_power.parquet")
    block3_parser.add_argument("--input-daily-flags-csv", required=True, help="Path to 02_sdt_daily_flags.csv")
    block3_parser.add_argument("--output-dir", required=True, help="Directory for Block 3 artifacts")
    block3_parser.add_argument(
        "--fit-mode",
        choices=["mask_to_nan", "filter_rows"],
        default="mask_to_nan",
        help="How to build fit time series",
    )
    block3_parser.add_argument("--min-fit-days", type=int, default=10, help="Minimum required fit days")

    block4_parser = sub.add_parser("run-block4", help="Run daily peak quantile normalization (Block 4)")
    block4_parser.add_argument("--input-power-fit-parquet", required=True, help="Path to 05_power_fit.parquet")
    block4_parser.add_argument("--output-dir", required=True, help="Directory for Block 4 artifacts")
    block4_parser.add_argument("--quantile", type=float, default=0.995, help="Daily peak quantile")
    block4_parser.add_argument(
        "--min-fit-samples-day",
        type=int,
        default=1,
        help="Minimum non-null fit samples required per day",
    )
    dropna_group = block4_parser.add_mutually_exclusive_group()
    dropna_group.add_argument(
        "--dropna-output",
        dest="dropna_output",
        action="store_true",
        help="Write only non-null normalized samples (default)",
    )
    dropna_group.add_argument(
        "--keep-nan-output",
        dest="dropna_output",
        action="store_false",
        help="Keep full timeline including NaN normalized samples",
    )
    block4_parser.set_defaults(dropna_output=True)

    block5_parser = sub.add_parser("run-block5", help="Run orientation estimation (single/two-plane)")
    block5_parser.add_argument("--input-p-norm-parquet", required=True, help="Path to 07_p_norm_clear.parquet")
    block5_parser.add_argument("--output-dir", required=True, help="Directory for Block 5 artifacts")
    block5_parser.add_argument("--latitude", required=True, type=float, help="Latitude in degrees")
    block5_parser.add_argument("--longitude", required=True, type=float, help="Longitude in degrees")
    block5_parser.add_argument("--timezone", default=None, help="Optional timezone override for naive timestamps")
    block5_parser.add_argument("--tilt-step", type=int, default=5, help="Coarse tilt step in degrees")
    block5_parser.add_argument("--az-step", type=int, default=10, help="Coarse azimuth step in degrees")
    block5_parser.add_argument("--topk", type=int, default=20, help="Number of top candidates to export")
    block5_parser.add_argument("--quantile", type=float, default=0.995, help="Daily normalization quantile")
    block5_parser.add_argument(
        "--norm-mode",
        choices=["quantile", "max"],
        default="quantile",
        help="Daily model normalization mode",
    )
    block5_parser.add_argument(
        "--two-plane-delta-az-deg",
        type=float,
        default=90.0,
        help="Two-plane half-delta in azimuth degrees (east=center-delta, west=center+delta)",
    )
    block5_parser.add_argument(
        "--skip-two-plane",
        action="store_true",
        help="Skip two-plane fitting and keep single-plane result",
    )
    block5_parser.add_argument(
        "--two-plane-if-rmse-ge",
        type=float,
        default=0.0,
        help="Run two-plane only if best single-plane RMSE is >= threshold (0.0 means always run)",
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

    if args.command == "run-block2":
        fix_shifts = not args.no_fix_shifts
        if args.input_parquet:
            result = run_block2_sdt_from_parquet(
                input_parquet=args.input_parquet,
                output_dir=args.output_dir,
                solver=args.solver,
                fix_shifts=fix_shifts,
                power_col=args.power_col,
            )
        else:
            result = run_block2_sdt_from_csv(
                input_csv=args.input_csv,
                output_dir=args.output_dir,
                timestamp_col=args.timestamp_col,
                power_col=args.csv_power_col,
                timezone=args.timezone,
                resample_if_irregular=not args.no_resample,
                min_samples=args.min_samples,
                clip_negative_power=not args.keep_negative_power,
                solver=args.solver,
                fix_shifts=fix_shifts,
            )

        print(
            json.dumps(
                {
                    "status": result.status,
                    "solver": result.solver,
                    "fix_shifts": result.fix_shifts,
                    "has_report": result.report is not None,
                    "has_daily_flags": result.daily_flags is not None and not result.daily_flags.empty,
                    "has_raw_data_matrix": result.raw_data_matrix is not None,
                    "has_filled_data_matrix": result.filled_data_matrix is not None,
                    "has_error": result.error is not None,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "run-block3":
        result = run_block3_from_files(
            input_power_parquet=args.input_power_parquet,
            input_daily_flags_csv=args.input_daily_flags_csv,
            output_dir=args.output_dir,
            fit_mode=args.fit_mode,
            min_fit_days=args.min_fit_days,
        )
        print(
            json.dumps(
                {
                    "status": result.status,
                    "n_fit_days": result.n_fit_days,
                    "min_required_fit_days": result.min_required_fit_days,
                    "n_days_total": result.n_days_total,
                    "n_unmatched_days": result.n_unmatched_days,
                    "fit_mode": result.fit_mode,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "run-block4":
        result = run_block4_from_files(
            input_power_fit_parquet=args.input_power_fit_parquet,
            output_dir=args.output_dir,
            quantile=args.quantile,
            min_fit_samples_day=args.min_fit_samples_day,
            dropna_output=args.dropna_output,
        )
        print(
            json.dumps(
                {
                    "quantile_used": result.quantile,
                    "min_fit_samples_day": result.min_fit_samples_day,
                    "dropna_output": result.dropna_output,
                    "n_samples_norm_total": int(result.diagnostics.get("n_samples_norm_total", 0)),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "run-block5":
        result = run_block5_from_files(
            input_p_norm_parquet=args.input_p_norm_parquet,
            output_dir=args.output_dir,
            latitude=args.latitude,
            longitude=args.longitude,
            timezone=args.timezone,
            tilt_step=args.tilt_step,
            az_step=args.az_step,
            topk=args.topk,
            quantile=args.quantile,
            norm_mode=args.norm_mode,
            two_plane_delta_az_deg=args.two_plane_delta_az_deg,
            skip_two_plane=args.skip_two_plane,
            two_plane_if_rmse_ge=args.two_plane_if_rmse_ge,
        )
        print(json.dumps(result, indent=2))
        t = result.get("timing_seconds", {})
        print(
            f"Block5 done in {t.get('total', 0.0):.2f} s "
            f"(precompute={t.get('precompute', 0.0):.2f}, "
            f"coarse={t.get('coarse_single', 0.0):.2f}, "
            f"fine={t.get('fine_single', 0.0):.2f}, "
            f"two_plane={t.get('coarse_two_plane', 0.0):.2f})"
        )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
