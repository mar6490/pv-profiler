from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser(description="Plot benchmark results")
    p.add_argument("--input", required=True, help="Path to benchmark_results.csv")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out_dir = Path(args.input).resolve().parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    two = df[df.get("model_type", pd.Series(dtype=str)) == "two_plane"].copy()
    if not two.empty and {"weight_true", "weight_est"}.issubset(two.columns):
        fig, ax = plt.subplots()
        ax.scatter(two["weight_true"], two["weight_est"], s=12)
        ax.set_xlabel("weight_true")
        ax.set_ylabel("weight_est")
        ax.set_title("Weight true vs estimated")
        fig.tight_layout()
        fig.savefig(out_dir / "scatter_weight_true_vs_est.png", dpi=150)
        plt.close(fig)

    if not two.empty and {"center_true", "center_est"}.issubset(two.columns):
        fig, ax = plt.subplots()
        ax.scatter(two["center_true"], two["center_est"], s=12)
        ax.set_xlabel("center_true")
        ax.set_ylabel("center_est")
        ax.set_title("Center true vs estimated")
        fig.tight_layout()
        fig.savefig(out_dir / "scatter_center_true_vs_est.png", dpi=150)
        plt.close(fig)

    if {"azimuth_true", "az_circular_err_deg"}.issubset(df.columns):
        fig, ax = plt.subplots()
        ax.scatter(df["azimuth_true"], df["az_circular_err_deg"], s=12)
        ax.set_xlabel("azimuth_true")
        ax.set_ylabel("az_err_deg")
        ax.set_title("Azimuth true vs circular error")
        fig.tight_layout()
        fig.savefig(out_dir / "scatter_az_true_vs_az_err.png", dpi=150)
        plt.close(fig)

    if {"tilt_true", "tilt_abs_err"}.issubset(df.columns):
        fig, ax = plt.subplots()
        ax.scatter(df["tilt_true"], df["tilt_abs_err"], s=12)
        ax.set_xlabel("tilt_true")
        ax.set_ylabel("tilt_err_deg")
        ax.set_title("Tilt true vs abs error")
        fig.tight_layout()
        fig.savefig(out_dir / "scatter_tilt_true_vs_tilt_err.png", dpi=150)
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
