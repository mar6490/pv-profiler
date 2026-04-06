#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAPER_RC = {
    "figure.figsize": (6.5, 4.5),
    "font.size": 13,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.grid": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PV orientation batch outputs against ground truth.")
    parser.add_argument("--batch-summary", required=True, help="Path to batch_summary.csv")
    parser.add_argument("--metadata", required=True, help="Path to metadata.csv")
    parser.add_argument("--out-dir", default="outputs_eval", help="Output directory for metrics and plots")
    parser.add_argument("--dpi", type=int, default=300, help="Output figure DPI")
    return parser.parse_args()


def circular_diff_deg(a: pd.Series | np.ndarray | float, b: pd.Series | np.ndarray | float):
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    d = np.abs((a_arr - b_arr + 180.0) % 360.0 - 180.0)
    if np.isscalar(a) and np.isscalar(b):
        return float(d)
    return d


def has_cols(df: pd.DataFrame, cols: list[str]) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        warnings.warn(f"Missing columns {missing}; dependent metrics/plots will be skipped.")
        return False
    return True


def infer_system_type(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unknown"
    s = str(value).strip().lower()
    if "east-west" in s or "east_west" in s or "eastwest" in s:
        return "east-west"
    if s == "single":
        return "single"
    return s if s else "unknown"


def to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def metric_block(err: pd.Series, true_v: pd.Series | None = None, est_v: pd.Series | None = None, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    e = pd.to_numeric(err, errors="coerce").dropna()
    n = int(e.shape[0])
    out[f"{prefix}N"] = n
    if n == 0:
        out[f"{prefix}MAE"] = np.nan
        out[f"{prefix}RMSE"] = np.nan
        out[f"{prefix}MedianAE"] = np.nan
        out[f"{prefix}P90AE"] = np.nan
        out[f"{prefix}PearsonR"] = np.nan
        return out

    out[f"{prefix}MAE"] = float(e.mean())
    out[f"{prefix}RMSE"] = float(np.sqrt(np.mean(np.square(e))))
    out[f"{prefix}MedianAE"] = float(e.median())
    out[f"{prefix}P90AE"] = float(np.quantile(e, 0.90))

    pearson = np.nan
    if true_v is not None and est_v is not None:
        t = pd.to_numeric(true_v, errors="coerce")
        y = pd.to_numeric(est_v, errors="coerce")
        m = t.notna() & y.notna()
        if int(m.sum()) >= 2:
            pearson = float(np.corrcoef(t[m].to_numpy(), y[m].to_numpy())[0, 1])
    out[f"{prefix}PearsonR"] = pearson
    return out


def summarize_scores(df: pd.DataFrame, prefix: str = "") -> dict[str, float]:
    s = pd.to_numeric(df.get("score_rmse"), errors="coerce")
    out: dict[str, float] = {
        f"{prefix}score_rmse_mean": np.nan,
        f"{prefix}score_rmse_median": np.nan,
        f"{prefix}score_rmse_p90": np.nan,
    }
    s = s.dropna()
    if not s.empty:
        out[f"{prefix}score_rmse_mean"] = float(s.mean())
        out[f"{prefix}score_rmse_median"] = float(s.median())
        out[f"{prefix}score_rmse_p90"] = float(np.quantile(s, 0.90))
    return out


def save_fig(fig: plt.Figure, out_dir: Path, stem: str, dpi: int) -> None:
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_tilt(df: pd.DataFrame, out_dir: Path, dpi: int) -> bool:
    m = df["true_tilt"].notna() & df["est_tilt"].notna()
    if int(m.sum()) == 0:
        warnings.warn("No valid data for tilt scatter; skipping plot.")
        return False

    fig, ax = plt.subplots()
    ax.scatter(df.loc[m, "true_tilt"], df.loc[m, "est_tilt"], s=22, alpha=0.8, label="systems")
    lo = float(min(df.loc[m, "true_tilt"].min(), df.loc[m, "est_tilt"].min()))
    hi = float(max(df.loc[m, "true_tilt"].max(), df.loc[m, "est_tilt"].max()))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.2, label="y=x")
    ax.set_xlabel("True tilt (deg)")
    ax.set_ylabel("Estimated tilt (deg)")
    ax.set_title("Tilt: True vs Estimated")
    ax.legend()
    save_fig(fig, out_dir, "01_scatter_tilt", dpi)
    return True


def plot_scatter_azimuth_single(df: pd.DataFrame, out_dir: Path, dpi: int) -> bool:
    m = (df["system_type_norm"] == "single") & df["true_az_single"].notna() & df["est_az"].notna()
    if int(m.sum()) == 0:
        warnings.warn("No valid single-system data for azimuth scatter; skipping plot.")
        return False

    fig, ax = plt.subplots()
    ax.scatter(df.loc[m, "true_az_single"], df.loc[m, "est_az"], s=22, alpha=0.8, label="single systems")
    ax.plot([0, 360], [0, 360], color="black", linewidth=1.2, label="y=x")
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 360)
    ax.set_xlabel("True azimuth (deg)")
    ax.set_ylabel("Estimated azimuth (deg)")
    ax.set_title("Azimuth: True vs Estimated (single systems)")
    ax.legend()
    save_fig(fig, out_dir, "02_scatter_azimuth_single", dpi)
    return True


def plot_hist(df: pd.DataFrame, col: str, title: str, xlabel: str, stem: str, out_dir: Path, dpi: int) -> bool:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        warnings.warn(f"No valid data for histogram {stem}; skipping.")
        return False
    fig, ax = plt.subplots()
    bins = max(10, min(40, int(np.sqrt(len(s)) * 2)))
    ax.hist(s, bins=bins, edgecolor="white", alpha=0.9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    save_fig(fig, out_dir, stem, dpi)
    return True


def plot_heatmap_single(df: pd.DataFrame, out_dir: Path, dpi: int) -> bool:
    m = (df["system_type_norm"] == "single") & df["true_az_single"].notna() & df["true_tilt"].notna() & df["az_error_overall"].notna()
    d = df.loc[m, ["true_az_single", "true_tilt", "az_error_overall"]].copy()
    if d.empty:
        warnings.warn("No valid single-system data for heatmap; skipping.")
        return False

    az_bins = np.arange(0, 361, 30)  # 12 bins
    y_min = float(np.floor(d["true_tilt"].min() / 5) * 5)
    y_max = float(np.ceil(d["true_tilt"].max() / 5) * 5)
    if y_max <= y_min:
        y_max = y_min + 30
    tilt_bins = np.linspace(y_min, y_max, 7)  # 6 bins

    az_idx = np.digitize(d["true_az_single"], az_bins) - 1
    tilt_idx = np.digitize(d["true_tilt"], tilt_bins) - 1

    heat = np.full((len(tilt_bins) - 1, len(az_bins) - 1), np.nan)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            m_ij = (tilt_idx == i) & (az_idx == j)
            if np.any(m_ij):
                heat[i, j] = float(d.loc[m_ij, "az_error_overall"].mean())

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    im = ax.imshow(heat, aspect="auto", origin="lower")
    ax.set_title("Mean azimuth error by true tilt/azimuth bins (single)")
    ax.set_xlabel("True azimuth bin (deg)")
    ax.set_ylabel("True tilt bin (deg)")

    az_labels = [f"{int(az_bins[i])}-{int(az_bins[i+1])}" for i in range(len(az_bins) - 1)]
    tilt_labels = [f"{tilt_bins[i]:.1f}-{tilt_bins[i+1]:.1f}" for i in range(len(tilt_bins) - 1)]
    ax.set_xticks(np.arange(len(az_labels)))
    ax.set_xticklabels(az_labels, rotation=90)
    ax.set_yticks(np.arange(len(tilt_labels)))
    ax.set_yticklabels(tilt_labels)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean azimuth error (deg)")

    save_fig(fig, out_dir, "05_heatmap_az_error_by_true_tilt_az", dpi)
    return True


def plot_polar(df: pd.DataFrame, out_dir: Path, dpi: int) -> tuple[bool, bool]:
    m_true = (df["system_type_norm"] == "single") & df["true_az_single"].notna() & df["true_tilt"].notna()
    m_est = (df["system_type_norm"] == "single") & df["est_az"].notna() & df["est_tilt"].notna()

    did_true = False
    did_est = False

    if int(m_true.sum()) > 0:
        fig = plt.figure(figsize=(6.5, 5.2))
        ax = fig.add_subplot(111, projection="polar")
        ax.scatter(np.deg2rad(df.loc[m_true, "true_az_single"]), df.loc[m_true, "true_tilt"], s=18, alpha=0.65)
        ax.set_title("Polar view: True orientation (single systems)")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(225)
        save_fig(fig, out_dir, "06_polar_true", dpi)
        did_true = True
    else:
        warnings.warn("No valid single-system true data for polar true plot; skipping.")

    if int(m_est.sum()) > 0:
        fig = plt.figure(figsize=(6.5, 5.2))
        ax = fig.add_subplot(111, projection="polar")
        ax.scatter(np.deg2rad(df.loc[m_est, "est_az"]), df.loc[m_est, "est_tilt"], s=18, alpha=0.65)
        ax.set_title("Polar view: Estimated orientation (single systems)")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(225)
        save_fig(fig, out_dir, "07_polar_est", dpi)
        did_est = True
    else:
        warnings.warn("No valid single-system estimated data for polar est plot; skipping.")

    return did_true, did_est


def build_per_system_errors(merged: pd.DataFrame) -> pd.DataFrame:
    df = merged.copy()

    # canonical estimation columns
    df = to_numeric(df, ["tilt_deg", "azimuth_deg", "score_rmse", "score_bic"])
    df["est_tilt"] = df.get("tilt_deg")
    df["est_az"] = df.get("azimuth_deg")

    # choose available ground-truth tilt preference
    if "tilt" in df.columns:
        df["true_tilt"] = pd.to_numeric(df["tilt"], errors="coerce")
    else:
        df["true_tilt"] = np.nan
    if "tilt_deg_true" in df.columns:
        df["true_tilt"] = df["true_tilt"].where(df["true_tilt"].notna(), pd.to_numeric(df["tilt_deg_true"], errors="coerce"))

    # single-system azimuth GT
    df["true_az_single"] = pd.to_numeric(df.get("azimuth_deg_true"), errors="coerce") if "azimuth_deg_true" in df.columns else np.nan

    # east-west references
    df["true_az_center"] = pd.to_numeric(df.get("azimuth_center_deg_true"), errors="coerce") if "azimuth_center_deg_true" in df.columns else np.nan
    df["true_az_east"] = pd.to_numeric(df.get("azimuth_east_deg_true"), errors="coerce") if "azimuth_east_deg_true" in df.columns else np.nan
    df["true_az_west"] = pd.to_numeric(df.get("azimuth_west_deg_true"), errors="coerce") if "azimuth_west_deg_true" in df.columns else np.nan

    df["system_type_norm"] = df.get("system_type", pd.Series(["unknown"] * len(df))).map(infer_system_type)

    df["tilt_error"] = np.abs(df["est_tilt"] - df["true_tilt"])

    # azimuth errors
    df["az_error_center"] = np.nan
    m_center = df["est_az"].notna() & df["true_az_center"].notna()
    if int(m_center.sum()) > 0:
        df.loc[m_center, "az_error_center"] = circular_diff_deg(df.loc[m_center, "est_az"], df.loc[m_center, "true_az_center"])

    df["az_error_min_plane"] = np.nan
    m_plane = df["est_az"].notna() & df["true_az_east"].notna() & df["true_az_west"].notna()
    if int(m_plane.sum()) > 0:
        e = circular_diff_deg(df.loc[m_plane, "est_az"], df.loc[m_plane, "true_az_east"])
        w = circular_diff_deg(df.loc[m_plane, "est_az"], df.loc[m_plane, "true_az_west"])
        df.loc[m_plane, "az_error_min_plane"] = np.minimum(e, w)

    df["az_error_overall"] = np.nan

    # single systems: compare against true_az_single
    m_single = (df["system_type_norm"] == "single") & df["est_az"].notna() & df["true_az_single"].notna()
    if int(m_single.sum()) > 0:
        df.loc[m_single, "az_error_overall"] = circular_diff_deg(df.loc[m_single, "est_az"], df.loc[m_single, "true_az_single"])

    # east-west systems: prefer center, else min_plane
    m_ew = df["system_type_norm"].astype(str).str.contains("east-west", case=False, na=False)
    m_ew_center = m_ew & df["az_error_center"].notna()
    df.loc[m_ew_center, "az_error_overall"] = df.loc[m_ew_center, "az_error_center"]
    m_ew_fallback = m_ew & df["az_error_overall"].isna() & df["az_error_min_plane"].notna()
    df.loc[m_ew_fallback, "az_error_overall"] = df.loc[m_ew_fallback, "az_error_min_plane"]

    cols = [
        "system_id",
        "system_type",
        "true_tilt",
        "est_tilt",
        "tilt_error",
        "true_az_single",
        "true_az_center",
        "true_az_east",
        "true_az_west",
        "est_az",
        "az_error_overall",
        "az_error_center",
        "az_error_min_plane",
        "score_rmse",
        "score_bic",
        "system_type_norm",
    ]
    return df[cols]


def compute_metrics(errors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = {}
    overall.update(metric_block(errors["tilt_error"], errors["true_tilt"], errors["est_tilt"], prefix="tilt_"))
    overall.update(metric_block(errors["az_error_overall"], errors["true_az_single"], errors["est_az"], prefix="azimuth_"))
    overall.update(summarize_scores(errors, prefix=""))
    overall_df = pd.DataFrame([overall])

    rows = []
    for stype, part in errors.groupby("system_type_norm", dropna=False):
        row = {"system_type": stype}
        row.update(metric_block(part["tilt_error"], part["true_tilt"], part["est_tilt"], prefix="tilt_"))
        row.update(metric_block(part["az_error_overall"], part["true_az_single"], part["est_az"], prefix="azimuth_"))
        row.update(summarize_scores(part, prefix=""))
        rows.append(row)
    by_type_df = pd.DataFrame(rows).sort_values("system_type").reset_index(drop=True)

    return overall_df, by_type_df


def write_summary_txt(out_dir: Path, overall: pd.DataFrame, by_type: pd.DataFrame, n_merged: int, n_input: int) -> None:
    o = overall.iloc[0].to_dict() if not overall.empty else {}
    lines = [
        "PV Batch Evaluation Summary",
        "===========================",
        f"Input systems in batch summary: {n_input}",
        f"Systems after metadata inner join: {n_merged}",
        "",
        "Overall metrics:",
        f"- Tilt MAE / RMSE / P90: {o.get('tilt_MAE', np.nan):.3f} / {o.get('tilt_RMSE', np.nan):.3f} / {o.get('tilt_P90AE', np.nan):.3f}",
        f"- Azimuth MAE / RMSE / P90: {o.get('azimuth_MAE', np.nan):.3f} / {o.get('azimuth_RMSE', np.nan):.3f} / {o.get('azimuth_P90AE', np.nan):.3f}",
        f"- Score RMSE mean / median / p90: {o.get('score_rmse_mean', np.nan):.4f} / {o.get('score_rmse_median', np.nan):.4f} / {o.get('score_rmse_p90', np.nan):.4f}",
        "",
        "By system type:",
    ]
    if by_type.empty:
        lines.append("- no grouped metrics available")
    else:
        for _, r in by_type.iterrows():
            lines.append(
                f"- {r.get('system_type')}: tilt_MAE={r.get('tilt_MAE', np.nan):.3f}, "
                f"azimuth_MAE={r.get('azimuth_MAE', np.nan):.3f}, score_rmse_mean={r.get('score_rmse_mean', np.nan):.4f}"
            )

    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(PAPER_RC)

    batch = pd.read_csv(args.batch_summary)
    meta = pd.read_csv(args.metadata)

    if "system_id" not in batch.columns:
        raise ValueError("batch_summary.csv must contain column 'system_id'")
    if "system_id" not in meta.columns:
        raise ValueError("metadata.csv must contain column 'system_id'")

    # status filter (if available)
    if "status" in batch.columns:
        batch_use = batch.loc[batch["status"].astype(str).str.lower() == "ok"].copy()
    else:
        batch_use = batch.copy()

    if batch_use.empty:
        raise ValueError("No rows to evaluate after status filter (status == 'ok').")

    batch_use["system_id"] = pd.to_numeric(batch_use["system_id"], errors="coerce")
    meta["system_id"] = pd.to_numeric(meta["system_id"], errors="coerce")

    batch_use = batch_use.dropna(subset=["system_id"]).copy()
    meta = meta.dropna(subset=["system_id"]).copy()

    batch_use["system_id"] = batch_use["system_id"].astype(int)
    meta["system_id"] = meta["system_id"].astype(int)

    merged = batch_use.merge(meta, on="system_id", how="inner", suffixes=("", "_meta"))
    if merged.empty:
        raise ValueError("No overlapping system_id values between batch_summary and metadata (inner join empty).")

    errors = build_per_system_errors(merged)

    per_system_path = out_dir / "per_system_errors.csv"
    errors.drop(columns=["system_type_norm"], errors="ignore").to_csv(per_system_path, index=False)

    overall_df, by_type_df = compute_metrics(errors)
    overall_df.to_csv(out_dir / "metrics_overall.csv", index=False)
    by_type_df.to_csv(out_dir / "metrics_by_system_type.csv", index=False)

    # bonus: worst 5 systems by azimuth error
    worst = errors[["system_id", "system_type", "az_error_overall", "tilt_error", "score_rmse", "score_bic"]].copy()
    worst = worst.sort_values("az_error_overall", ascending=False, na_position="last").head(5)
    worst.to_csv(out_dir / "worst_5_systems.csv", index=False)

    # plotting (graceful skips)
    _ = plot_scatter_tilt(errors, out_dir, args.dpi)
    _ = plot_scatter_azimuth_single(errors, out_dir, args.dpi)
    _ = plot_hist(errors, "tilt_error", "Tilt absolute error distribution", "Tilt error (deg)", "03_hist_tilt_error", out_dir, args.dpi)
    _ = plot_hist(errors, "az_error_overall", "Azimuth circular error distribution", "Azimuth error (deg)", "04_hist_az_error", out_dir, args.dpi)
    _ = plot_heatmap_single(errors, out_dir, args.dpi)
    _ = plot_polar(errors, out_dir, args.dpi)

    write_summary_txt(
        out_dir=out_dir,
        overall=overall_df,
        by_type=by_type_df,
        n_merged=len(merged),
        n_input=len(batch_use),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
