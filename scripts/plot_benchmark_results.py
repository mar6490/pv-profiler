from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _has_cols(df: pd.DataFrame, cols: set[str]) -> bool:
    return cols.issubset(set(df.columns))


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _plot_scatter(df: pd.DataFrame, x: str, y: str, xlabel: str, ylabel: str, title: str, out_path: Path) -> None:
    d = pd.DataFrame({x: _to_num(df[x]), y: _to_num(df[y])}).dropna()
    if d.empty:
        print(f"[plot_benchmark_results] skip {out_path.name}: no valid rows")
        return
    fig, ax = plt.subplots()
    ax.scatter(d[x], d[y], s=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Plot benchmark results")
    p.add_argument("--input", required=True, help="Path to benchmark_results.csv")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out_dir = Path(args.input).resolve().parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    two = df[df.get("model_type", pd.Series(dtype=str)).astype(str) == "two_plane"].copy()

    if not two.empty and _has_cols(two, {"weight_true", "weight_est"}):
        _plot_scatter(
            two,
            "weight_true",
            "weight_est",
            "weight_true",
            "weight_est",
            "Weight true vs estimated",
            out_dir / "scatter_weight_true_vs_est.png",
        )
    else:
        print("[plot_benchmark_results] skip scatter_weight_true_vs_est.png: missing columns or no two_plane rows")

    if not two.empty and _has_cols(two, {"center_true", "center_est"}):
        _plot_scatter(
            two,
            "center_true",
            "center_est",
            "center_true",
            "center_est",
            "Center true vs estimated",
            out_dir / "scatter_center_true_vs_est.png",
        )
    else:
        print("[plot_benchmark_results] skip scatter_center_true_vs_est.png: missing columns or no two_plane rows")

    if _has_cols(df, {"azimuth_true", "az_circular_err_deg"}):
        _plot_scatter(
            df,
            "azimuth_true",
            "az_circular_err_deg",
            "azimuth_true",
            "az_err_deg",
            "Azimuth true vs circular error",
            out_dir / "scatter_az_true_vs_az_err.png",
        )
    else:
        print("[plot_benchmark_results] skip scatter_az_true_vs_az_err.png: missing columns")

    if _has_cols(df, {"tilt_true", "tilt_abs_err"}):
        _plot_scatter(
            df,
            "tilt_true",
            "tilt_abs_err",
            "tilt_true",
            "tilt_err_deg",
            "Tilt true vs abs error",
            out_dir / "scatter_tilt_true_vs_tilt_err.png",
        )
    else:
        print("[plot_benchmark_results] skip scatter_tilt_true_vs_tilt_err.png: missing columns")

    # EW-specific optional plots
    if _has_cols(df, {"center_true", "az_center_abs_err_deg"}):
        _plot_scatter(
            df,
            "center_true",
            "az_center_abs_err_deg",
            "center_true",
            "center_err_deg",
            "Center true vs center azimuth error",
            out_dir / "scatter_center_true_vs_center_err.png",
        )
    else:
        print("[plot_benchmark_results] skip scatter_center_true_vs_center_err.png: missing columns")

    if _has_cols(df, {"tilt_true", "az_plane_abs_err_deg"}):
        _plot_scatter(
            df,
            "tilt_true",
            "az_plane_abs_err_deg",
            "tilt_true",
            "plane_err_deg",
            "Tilt true vs east/west plane azimuth error",
            out_dir / "scatter_tilt_true_vs_plane_err.png",
        )
    else:
        print("[plot_benchmark_results] skip scatter_tilt_true_vs_plane_err.png: missing columns")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
