from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def circular_err_deg(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> np.ndarray:
    aa = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    bb = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
    d = np.abs((aa - bb + 180.0) % 360.0 - 180.0)
    return d


def fold180_err_deg(e: pd.Series | np.ndarray) -> np.ndarray:
    ee = pd.to_numeric(e, errors="coerce").to_numpy(dtype=float)
    return np.minimum(ee, 180.0 - ee)


def _has_cols(df: pd.DataFrame, cols: set[str]) -> bool:
    return cols.issubset(set(df.columns))


def _pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _plot_scatter(df: pd.DataFrame, x: str, y: str, xlabel: str, ylabel: str, title: str, out_path: Path) -> None:
    d = pd.DataFrame({x: pd.to_numeric(df[x], errors="coerce"), y: pd.to_numeric(df[y], errors="coerce")}).dropna()
    if d.empty:
        print(f"[plot_benchmark_results] skip {out_path.name}: no valid rows")
        return
    fig, ax = plt.subplots()
    ax.scatter(d[x], d[y], s=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hist(series: pd.Series, title: str, out_path: Path, xlabel: str) -> None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        print(f"[plot_benchmark_results] skip {out_path.name}: no valid rows")
        return
    fig, ax = plt.subplots()
    bins = max(8, min(40, int(np.sqrt(len(s)) * 2)))
    ax.hist(s, bins=bins)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_box_by_model(df: pd.DataFrame, value_col: str, out_path: Path, ylabel: str) -> None:
    if not _has_cols(df, {"model_type", value_col}):
        print(f"[plot_benchmark_results] skip {out_path.name}: missing columns")
        return

    parts: list[pd.Series] = []
    labels: list[str] = []
    for m in ["single", "two_plane"]:
        vals = pd.to_numeric(df.loc[df["model_type"].astype(str) == m, value_col], errors="coerce").dropna()
        if not vals.empty:
            parts.append(vals)
            labels.append(m)

    if len(parts) < 2:
        print(f"[plot_benchmark_results] skip {out_path.name}: need both single and two_plane with valid rows")
        return

    fig, ax = plt.subplots()
    ax.boxplot(parts, labels=labels)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by model type")
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

    # Robust error column resolution
    tilt_err_col = _pick_first_existing(df, ["tilt_abs_err", "tilt_abs_err_deg"])

    # two-plane subset for EW-focused plots
    two = df[df.get("model_type", pd.Series(dtype=str)).astype(str) == "two_plane"].copy()

    # If center error column missing, derive from center_true/center_est.
    if "az_center_abs_err_deg" not in two.columns and _has_cols(two, {"center_true", "center_est"}):
        two["az_center_abs_err_deg"] = circular_err_deg(two["center_est"], two["center_true"])

    if "az_center_abs_err_deg" in two.columns:
        # EW center is identifiable only modulo 180°; fold is the primary meaningful center error.
        two["center_err_fold180_deg"] = fold180_err_deg(two["az_center_abs_err_deg"])

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

    if not two.empty and "weight_abs_err" in two.columns:
        plot_hist(
            two["weight_abs_err"],
            "Weight absolute error (two_plane)",
            out_dir / "hist_weight_abs_err.png",
            "weight abs error",
        )
    else:
        print("[plot_benchmark_results] skip hist_weight_abs_err.png: missing columns or no two_plane rows")

    if not two.empty and _has_cols(two, {"center_true", "center_est"}):
        _plot_scatter(
            two,
            "center_true",
            "center_est",
            "center_true",
            "center_est",
            "Center true vs estimated (visual check)",
            out_dir / "scatter_center_true_vs_est.png",
        )
    else:
        print("[plot_benchmark_results] skip scatter_center_true_vs_est.png: missing columns or no two_plane rows")

    if not two.empty and _has_cols(two, {"center_true", "center_err_fold180_deg"}):
        _plot_scatter(
            two,
            "center_true",
            "center_err_fold180_deg",
            "center_true",
            "center_err_fold180_deg",
            "Center error folded modulo 180°",
            out_dir / "scatter_center_true_vs_center_err_fold180.png",
        )
    else:
        print("[plot_benchmark_results] skip scatter_center_true_vs_center_err_fold180.png: missing columns or no two_plane rows")

    if not two.empty and "center_err_fold180_deg" in two.columns:
        plot_hist(
            two["center_err_fold180_deg"],
            "Center error folded modulo 180° (two_plane)",
            out_dir / "hist_center_err_fold180.png",
            "center_err_fold180_deg",
        )
    else:
        print("[plot_benchmark_results] skip hist_center_err_fold180.png: missing columns or no two_plane rows")

    if not two.empty and "az_plane_abs_err_deg" in two.columns:
        plot_hist(
            two["az_plane_abs_err_deg"],
            "Plane azimuth error (two_plane)",
            out_dir / "hist_plane_err.png",
            "az_plane_abs_err_deg",
        )
    else:
        print("[plot_benchmark_results] skip hist_plane_err.png: missing columns or no two_plane rows")

    if not two.empty and _has_cols(two, {"center_err_fold180_deg", "az_plane_abs_err_deg"}):
        _plot_scatter(
            two,
            "center_err_fold180_deg",
            "az_plane_abs_err_deg",
            "center_err_fold180_deg",
            "az_plane_abs_err_deg",
            "Folded center error vs plane error",
            out_dir / "scatter_center_err_fold180_vs_plane_err.png",
        )
    else:
        print("[plot_benchmark_results] skip scatter_center_err_fold180_vs_plane_err.png: missing columns or no two_plane rows")

    # Single-only azimuth plot: do not mix EW rows.
    single = df[df.get("model_type", pd.Series(dtype=str)).astype(str) == "single"].copy()
    if not single.empty and _has_cols(single, {"azimuth_true", "az_circular_err_deg"}):
        _plot_scatter(
            single,
            "azimuth_true",
            "az_circular_err_deg",
            "azimuth_true",
            "az_err_deg",
            "Azimuth true vs circular error (single)",
            out_dir / "scatter_az_true_vs_az_err.png",
        )
    else:
        print("[plot_benchmark_results] skip scatter_az_true_vs_az_err.png: missing columns or no single rows")

    if tilt_err_col is not None and _has_cols(df, {"tilt_true", tilt_err_col}):
        _plot_scatter(
            df,
            "tilt_true",
            tilt_err_col,
            "tilt_true",
            "tilt_err_deg",
            "Tilt true vs abs error",
            out_dir / "scatter_tilt_true_vs_tilt_err.png",
        )
    else:
        print("[plot_benchmark_results] skip scatter_tilt_true_vs_tilt_err.png: missing columns")

    if tilt_err_col is not None:
        plot_box_by_model(df, tilt_err_col, out_dir / "box_tilt_abs_err_by_model.png", "tilt_abs_err")
    else:
        print("[plot_benchmark_results] skip box_tilt_abs_err_by_model.png: missing tilt error column")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
