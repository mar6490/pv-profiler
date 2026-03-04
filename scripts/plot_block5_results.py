from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot diagnostics from Block 5 orientation output")
    p.add_argument("--input-dir", required=True, help="Block 5 output directory")
    p.add_argument("--true-tilt", type=float, default=None, help="Optional true tilt marker")
    p.add_argument("--true-azimuth", type=float, default=None, help="Optional true azimuth marker")
    return p


def main() -> int:
    args = build_parser().parse_args()
    in_dir = Path(args.input_dir)

    single_grid = pd.read_csv(in_dir / "09a_orientation_single_full_grid.csv")
    profile = pd.read_csv(in_dir / "10_profile_compare.csv")

    png_heatmap = in_dir / "plot_rmse_heatmap.png"
    png_profile = in_dir / "plot_profile_compare.png"
    png_residual = in_dir / "plot_residual_vs_time.png"
    png_sens = in_dir / "plot_rmse_vs_azimuth.png"
    pdf_path = in_dir / "block5_diagnostics.pdf"

    figs: list[plt.Figure] = []

    # Plot 1: RMSE Heatmap
    pivot = single_grid.pivot(index="tilt_deg", columns="azimuth_deg", values="rmse").sort_index().sort_index(axis=1)
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    im = ax1.imshow(pivot.values, aspect="auto", origin="lower")
    ax1.set_xticks(range(len(pivot.columns)))
    ax1.set_xticklabels([f"{x:.0f}" for x in pivot.columns], rotation=90)
    ax1.set_yticks(range(len(pivot.index)))
    ax1.set_yticklabels([f"{x:.0f}" for x in pivot.index])
    ax1.set_xlabel("azimuth_deg")
    ax1.set_ylabel("tilt_deg")
    ax1.set_title("RMSE heatmap (single-plane)")
    fig1.colorbar(im, ax=ax1, label="RMSE")

    best = single_grid.sort_values("rmse").iloc[0]
    ix = list(pivot.columns).index(best["azimuth_deg"])
    iy = list(pivot.index).index(best["tilt_deg"])
    ax1.scatter(ix, iy, marker="x", s=80, label="best", c="white")

    if args.true_tilt is not None and args.true_azimuth is not None:
        az_vals = list(pivot.columns)
        tilt_vals = list(pivot.index)
        closest_x = min(range(len(az_vals)), key=lambda i: abs(az_vals[i] - args.true_azimuth))
        closest_y = min(range(len(tilt_vals)), key=lambda i: abs(tilt_vals[i] - args.true_tilt))
        ax1.scatter(closest_x, closest_y, marker="o", s=60, facecolors="none", edgecolors="red", label="true")
    ax1.legend(loc="upper right")
    fig1.tight_layout()
    fig1.savefig(png_heatmap, dpi=150)
    figs.append(fig1)

    # Plot 2: profile compare
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(profile["minute_of_day"], profile["observed_p_norm"], label="observed_p_norm")
    ax2.plot(profile["minute_of_day"], profile["predicted_p_norm"], label="predicted_p_norm")
    ax2.set_xlabel("minute_of_day")
    ax2.set_ylabel("p_norm")
    ax2.set_title("Observed vs Predicted Profile")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(png_profile, dpi=150)
    figs.append(fig2)

    # Plot 3: residual vs time
    residual = profile.copy()
    residual["residual"] = residual["observed_p_norm"] - residual["predicted_p_norm"]
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(residual["minute_of_day"], residual["residual"])
    ax3.axhline(0.0, color="black", linewidth=1)
    ax3.set_xlabel("minute_of_day")
    ax3.set_ylabel("residual")
    ax3.set_title("Residual vs Time of Day")
    fig3.tight_layout()
    fig3.savefig(png_residual, dpi=150)
    figs.append(fig3)

    # Plot 4: RMSE vs azimuth for best tilt
    best_tilt = float(single_grid.sort_values("rmse").iloc[0]["tilt_deg"])
    sens = single_grid[single_grid["tilt_deg"] == best_tilt].sort_values("azimuth_deg")
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    ax4.plot(sens["azimuth_deg"], sens["rmse"], marker="o")
    ax4.set_xlabel("azimuth_deg")
    ax4.set_ylabel("rmse")
    ax4.set_title(f"RMSE vs Azimuth (tilt={best_tilt:.0f}°)")
    fig4.tight_layout()
    fig4.savefig(png_sens, dpi=150)
    figs.append(fig4)

    with PdfPages(pdf_path) as pdf:
        for fig in figs:
            pdf.savefig(fig)

    for fig in figs:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
