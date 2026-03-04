from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KEY_FILES = [
    "00_status.json",
    "01_input_diagnostics.json",
    "02_sdt_daily_flags.csv",
    "05_power_fit.parquet",
    "07_p_norm_clear.parquet",
    "08_orientation_result.json",
    "09a_orientation_single_full_grid.csv",
    "09b_orientation_two_plane_full_grid.csv",
]


def _parse_system_id(name: str) -> int | None:
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else None


def _safe_read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_status_counts(df: pd.DataFrame, out: Path) -> None:
    s = df["status"].fillna("missing_status").value_counts()
    fig, ax = plt.subplots()
    s.plot(kind="bar", ax=ax)
    ax.set_ylabel("count")
    ax.set_title("Status counts")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_hist(df: pd.DataFrame, col: str, out: Path, title: str) -> None:
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    if vals.empty:
        return
    fig, ax = plt.subplots()
    ax.hist(vals, bins=max(8, min(40, int(len(vals) ** 0.5) * 2)))
    ax.set_title(title)
    ax.set_xlabel(col)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_presence_bar(presence: dict[str, bool], out: Path) -> None:
    names = list(presence.keys())
    vals = [1 if presence[n] else 0 for n in names]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(range(len(names)), vals)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("present (0/1)")
    ax.set_title("Artifact presence")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_daily_flags(flags_csv: Path, out: Path) -> None:
    if not flags_csv.exists():
        return
    df = pd.read_csv(flags_csv)
    bool_cols = [c for c in df.columns if pd.api.types.is_bool_dtype(df[c])]
    if not bool_cols:
        return
    counts = {c: int(df[c].fillna(False).sum()) for c in bool_cols}
    if not counts:
        return
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(list(counts.keys()), list(counts.values()))
    ax.set_ylabel("true count")
    ax.set_title("Daily flags")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _best_by(df: pd.DataFrame, col: str) -> pd.Series | None:
    if col not in df.columns or df.empty:
        return None
    vals = pd.to_numeric(df[col], errors="coerce")
    if vals.isna().all():
        return None
    return df.loc[vals.idxmin()]


def _metadata_true_cols(df: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
    tilt_candidates = ["tilt_true", "true_tilt", "tilt_deg_true"]
    center_candidates = ["center_true", "azimuth_center_true", "true_center", "az_center_true"]
    azimuth_candidates = ["azimuth_true", "true_azimuth", "az_true", "azimuth_deg_true"]
    tilt_col = next((c for c in tilt_candidates if c in df.columns), None)
    center_col = next((c for c in center_candidates if c in df.columns), None)
    azimuth_col = next((c for c in azimuth_candidates if c in df.columns), None)
    return tilt_col, center_col, azimuth_col


def _fold_center_0_180(v: float | int | None) -> float | None:
    if v is None or pd.isna(v):
        return None
    return float(v) % 180.0


def _plot_landscape(
    *,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    val_col: str,
    out: Path,
    title: str,
    true_x: float | None,
    true_y: float | None,
) -> None:
    required = {x_col, y_col, val_col}
    if not required.issubset(df.columns):
        return

    cols = [x_col, y_col, val_col, "rmse", "bic"]
    dd = df.loc[:, [c for c in dict.fromkeys(cols) if c in df.columns]].copy()
    dd[x_col] = pd.to_numeric(dd[x_col], errors="coerce")
    dd[y_col] = pd.to_numeric(dd[y_col], errors="coerce")
    dd[val_col] = pd.to_numeric(dd[val_col], errors="coerce")
    dd = dd.dropna(subset=[x_col, y_col, val_col])
    if dd.empty:
        return

    pivot = dd.pivot_table(index=y_col, columns=x_col, values=val_col, aggfunc="min").sort_index().sort_index(axis=1)
    if pivot.empty:
        return

    x_vals = pivot.columns.to_numpy(dtype=float)
    y_vals = pivot.index.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    mesh = ax.pcolormesh(x_vals, y_vals, pivot.to_numpy(dtype=float), shading="auto")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(val_col)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)

    best_rmse = _best_by(dd, "rmse")
    if best_rmse is not None:
        ax.scatter(
            [pd.to_numeric(best_rmse[x_col], errors="coerce")],
            [pd.to_numeric(best_rmse[y_col], errors="coerce")],
            marker="*",
            s=120,
            color="white",
            edgecolors="black",
            linewidths=0.8,
            label="best RMSE",
        )

    best_bic = _best_by(dd, "bic")
    if best_bic is not None:
        ax.scatter(
            [pd.to_numeric(best_bic[x_col], errors="coerce")],
            [pd.to_numeric(best_bic[y_col], errors="coerce")],
            marker="x",
            s=70,
            color="black",
            linewidths=1.2,
            label="best BIC",
        )

    if true_x is not None and true_y is not None:
        ax.scatter([true_x], [true_y], marker="o", s=80, facecolors="none", edgecolors="cyan", linewidths=1.6, label="true")

    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_1d_combo(
    *,
    df: pd.DataFrame,
    x_col: str,
    out: Path,
    title: str,
    true_x: float | None,
    x_label: str,
) -> None:
    if not {x_col, "rmse", "bic"}.issubset(df.columns):
        return
    d = df[[x_col, "rmse", "bic"]].copy()
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d["rmse"] = pd.to_numeric(d["rmse"], errors="coerce")
    d["bic"] = pd.to_numeric(d["bic"], errors="coerce")
    d = d.dropna(subset=[x_col, "rmse", "bic"])
    if d.empty:
        return

    rmse_by_x = d.groupby(x_col, as_index=True)["rmse"].min().sort_index()
    bic_by_x = d.groupby(x_col, as_index=True)["bic"].min().sort_index()
    if rmse_by_x.empty or bic_by_x.empty:
        return

    x_rmse = rmse_by_x.index.to_numpy(dtype=float)
    y_rmse = rmse_by_x.to_numpy(dtype=float)
    x_bic = bic_by_x.index.to_numpy(dtype=float)
    y_bic = bic_by_x.to_numpy(dtype=float)

    fig, ax1 = plt.subplots(figsize=(7, 4))
    l1 = ax1.plot(x_rmse, y_rmse, color="tab:blue", label="rmse_min")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("rmse_min", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    l2 = ax2.plot(x_bic, y_bic, color="tab:red", label="bic_min")
    ax2.set_ylabel("bic_min", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    best_rmse_x = float(x_rmse[np.argmin(y_rmse)])
    best_bic_x = float(x_bic[np.argmin(y_bic)])
    ax1.axvline(best_rmse_x, color="tab:blue", linestyle="--", linewidth=1.0, label="best RMSE")
    ax2.axvline(best_bic_x, color="tab:red", linestyle=":", linewidth=1.0, label="best BIC")

    if true_x is not None and not pd.isna(true_x):
        ax1.axvline(float(true_x), color="black", linestyle="-", linewidth=1.0, label="true")

    lines = l1 + l2
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="best")

    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _model_choice_color(s: pd.Series) -> np.ndarray:
    vals = s.astype(str)
    out = np.where(vals == "two_plane", 1, np.where(vals == "single", 0, -1))
    return out.astype(float)


def generate_diagnostics_v2(
    *,
    output_root: str | Path,
    systems_metadata_csv: str | Path | None = None,
    system_id_col: str = "system_id",
    health_plots: bool = False,
) -> pd.DataFrame:
    output_root = Path(output_root)
    diag_root = output_root / "diagnostics_v2"
    per_system_root = diag_root / "per_system"
    global_root = diag_root / "global"
    per_system_root.mkdir(parents=True, exist_ok=True)
    global_root.mkdir(parents=True, exist_ok=True)

    metadata_df: pd.DataFrame | None = None
    if systems_metadata_csv is not None:
        metadata_df = pd.read_csv(systems_metadata_csv)

    rows: list[dict] = []
    for system_dir in sorted(p for p in output_root.iterdir() if p.is_dir() and p.name != "diagnostics_v2"):
        system_id = _parse_system_id(system_dir.name)
        status_path = system_dir / "00_status.json"
        orient_path = system_dir / "08_orientation_result.json"
        single_grid_path = system_dir / "09a_orientation_single_full_grid.csv"
        two_grid_path = system_dir / "09b_orientation_two_plane_full_grid.csv"

        if not status_path.exists():
            print(f"[make-diagnostics] warning: missing {status_path}")
            continue

        status = _safe_read_json(status_path)
        orient = _safe_read_json(orient_path) if orient_path.exists() else {}
        if not orient_path.exists():
            print(f"[make-diagnostics] warning: missing {orient_path}")

        row = {
            "system_dir": system_dir.name,
            "system_id": system_id,
            "status": status.get("status"),
            "runtime_seconds": status.get("runtime_seconds"),
            "error_type": (status.get("error") or {}).get("type") if isinstance(status.get("error"), dict) else None,
            "error_message": (status.get("error") or {}).get("message") if isinstance(status.get("error"), dict) else None,
            "model_type": orient.get("model_type"),
            "tilt_deg": orient.get("tilt_deg"),
            "azimuth_deg": orient.get("azimuth_deg"),
            "azimuth_center_deg": orient.get("azimuth_center_deg"),
            "weight_east": orient.get("weight_east"),
            "score_rmse": orient.get("score_rmse"),
            "score_bic": orient.get("score_bic"),
            "best_single_rmse": np.nan,
            "best_single_bic": np.nan,
            "best_single_tilt_rmse": np.nan,
            "best_single_az_rmse": np.nan,
            "best_two_rmse": np.nan,
            "best_two_bic": np.nan,
            "best_two_tilt_rmse": np.nan,
            "best_two_center_rmse": np.nan,
            "delta_rmse": np.nan,
            "delta_bic": np.nan,
            "rmse_prefers_two_plane": np.nan,
            "bic_prefers_two_plane": np.nan,
        }
        for f in KEY_FILES:
            row[f"has_{f}"] = (system_dir / f).exists()

        true_tilt = None
        true_center = None
        true_azimuth = None
        if metadata_df is not None and system_id_col in metadata_df.columns and system_id is not None:
            m = metadata_df[pd.to_numeric(metadata_df[system_id_col], errors="coerce") == float(system_id)]
            if not m.empty:
                t_col, c_col, a_col = _metadata_true_cols(m)
                if t_col is not None:
                    true_tilt = pd.to_numeric(m.iloc[0][t_col], errors="coerce")
                    row["tilt_true"] = true_tilt
                if c_col is not None:
                    true_center = pd.to_numeric(m.iloc[0][c_col], errors="coerce")
                    row["center_true"] = true_center
                if a_col is not None:
                    true_azimuth = pd.to_numeric(m.iloc[0][a_col], errors="coerce")
                    row["azimuth_true"] = true_azimuth

        single_df = None
        if single_grid_path.exists():
            single_df = pd.read_csv(single_grid_path)
            best_single_rmse = _best_by(single_df, "rmse")
            best_single_bic = _best_by(single_df, "bic")
            if best_single_rmse is not None:
                row["best_single_rmse"] = pd.to_numeric(best_single_rmse.get("rmse"), errors="coerce")
                row["best_single_tilt_rmse"] = pd.to_numeric(best_single_rmse.get("tilt_deg"), errors="coerce")
                row["best_single_az_rmse"] = pd.to_numeric(best_single_rmse.get("azimuth_deg"), errors="coerce")
            if best_single_bic is not None:
                row["best_single_bic"] = pd.to_numeric(best_single_bic.get("bic"), errors="coerce")
        else:
            print(f"[make-diagnostics] warning: missing {single_grid_path}")

        two_df = None
        if two_grid_path.exists():
            two_df = pd.read_csv(two_grid_path)
            best_two_rmse = _best_by(two_df, "rmse")
            best_two_bic = _best_by(two_df, "bic")
            if best_two_rmse is not None:
                row["best_two_rmse"] = pd.to_numeric(best_two_rmse.get("rmse"), errors="coerce")
                row["best_two_tilt_rmse"] = pd.to_numeric(best_two_rmse.get("tilt_deg"), errors="coerce")
                row["best_two_center_rmse"] = pd.to_numeric(best_two_rmse.get("azimuth_center_deg"), errors="coerce")
            if best_two_bic is not None:
                row["best_two_bic"] = pd.to_numeric(best_two_bic.get("bic"), errors="coerce")
        else:
            print(f"[make-diagnostics] warning: missing {two_grid_path}")

        if pd.notna(row["best_single_rmse"]) and pd.notna(row["best_two_rmse"]):
            row["delta_rmse"] = float(row["best_two_rmse"] - row["best_single_rmse"])
            row["rmse_prefers_two_plane"] = bool(row["delta_rmse"] < 0)

        if pd.notna(row["best_single_bic"]) and pd.notna(row["best_two_bic"]):
            row["delta_bic"] = float(row["best_two_bic"] - row["best_single_bic"])
            row["bic_prefers_two_plane"] = bool(row["delta_bic"] < 0)

        sys_out = per_system_root / system_dir.name
        sys_out.mkdir(parents=True, exist_ok=True)
        if health_plots:
            _plot_presence_bar({f: (system_dir / f).exists() for f in KEY_FILES}, sys_out / "artifact_presence.png")
            _plot_daily_flags(system_dir / "02_sdt_daily_flags.csv", sys_out / "daily_flags.png")

        if single_df is not None:
            _plot_landscape(
                df=single_df,
                x_col="azimuth_deg",
                y_col="tilt_deg",
                val_col="rmse",
                out=sys_out / "rmse_single_landscape.png",
                title="Single-plane RMSE landscape",
                true_x=float(true_azimuth) if true_azimuth is not None and not pd.isna(true_azimuth) else None,
                true_y=float(true_tilt) if true_tilt is not None and not pd.isna(true_tilt) else None,
            )
            _plot_landscape(
                df=single_df,
                x_col="azimuth_deg",
                y_col="tilt_deg",
                val_col="bic",
                out=sys_out / "bic_single_landscape.png",
                title="Single-plane BIC landscape",
                true_x=float(true_azimuth) if true_azimuth is not None and not pd.isna(true_azimuth) else None,
                true_y=float(true_tilt) if true_tilt is not None and not pd.isna(true_tilt) else None,
            )
            _plot_1d_combo(
                df=single_df,
                x_col="azimuth_deg",
                out=sys_out / "single_1d_azimuth_rmse_bic.png",
                title="Single-plane min-over-tilt by azimuth",
                true_x=float(true_azimuth) if true_azimuth is not None and not pd.isna(true_azimuth) else None,
                x_label="azimuth_deg",
            )
        else:
            print(f"[make-diagnostics] warning: skipping single-plane landscapes/1D for {system_dir.name}")

        if two_df is not None:
            _plot_landscape(
                df=two_df,
                x_col="azimuth_center_deg",
                y_col="tilt_deg",
                val_col="rmse",
                out=sys_out / "rmse_two_plane_landscape.png",
                title="Two-plane RMSE landscape",
                true_x=_fold_center_0_180(true_center) if true_center is not None else None,
                true_y=float(true_tilt) if true_tilt is not None and not pd.isna(true_tilt) else None,
            )
            _plot_landscape(
                df=two_df,
                x_col="azimuth_center_deg",
                y_col="tilt_deg",
                val_col="bic",
                out=sys_out / "bic_two_plane_landscape.png",
                title="Two-plane BIC landscape",
                true_x=_fold_center_0_180(true_center) if true_center is not None else None,
                true_y=float(true_tilt) if true_tilt is not None and not pd.isna(true_tilt) else None,
            )
            _plot_1d_combo(
                df=two_df,
                x_col="azimuth_center_deg",
                out=sys_out / "two_plane_1d_center_rmse_bic.png",
                title="Two-plane min-over-tilt by center",
                true_x=_fold_center_0_180(true_center) if true_center is not None else None,
                x_label="azimuth_center_deg",
            )
        else:
            print(f"[make-diagnostics] warning: skipping two-plane landscapes/1D for {system_dir.name}")

        rows.append(row)

    summary = pd.DataFrame(rows)

    if metadata_df is not None and not summary.empty and system_id_col in metadata_df.columns:
        md = metadata_df.copy()
        md[system_id_col] = pd.to_numeric(md[system_id_col], errors="coerce").astype("Int64")
        summary["system_id"] = pd.to_numeric(summary["system_id"], errors="coerce").astype("Int64")
        summary = summary.merge(md, left_on="system_id", right_on=system_id_col, how="left")

    summary.to_csv(diag_root / "aggregated_metrics.csv", index=False)

    if not summary.empty:
        _plot_hist(summary, "delta_rmse", global_root / "hist_delta_rmse.png", "Delta RMSE (two - single)")
        _plot_hist(summary, "delta_bic", global_root / "hist_delta_bic.png", "Delta BIC (two - single)")

        dd = summary[["delta_rmse", "delta_bic"]].apply(pd.to_numeric, errors="coerce").dropna()
        if not dd.empty:
            fig, ax = plt.subplots()
            ax.scatter(dd["delta_rmse"], dd["delta_bic"], s=15)
            ax.axhline(0.0, color="black", linewidth=1)
            ax.axvline(0.0, color="black", linewidth=1)
            ax.set_xlabel("delta_rmse")
            ax.set_ylabel("delta_bic")
            ax.set_title("Delta RMSE vs Delta BIC")
            fig.tight_layout()
            fig.savefig(global_root / "scatter_delta_rmse_vs_delta_bic.png", dpi=150)
            plt.close(fig)

        center_col = next((c for c in ["center_true", "center_true_x", "center_true_y"] if c in summary.columns), None)
        tilt_col = next((c for c in ["tilt_true", "tilt_true_x", "tilt_true_y"] if c in summary.columns), None)
        if center_col is not None and tilt_col is not None and "model_type" in summary.columns:
            cc = summary.copy()
            cc["center_folded_0_180"] = cc[center_col].map(_fold_center_0_180)
            cc = cc.dropna(subset=["center_folded_0_180", tilt_col, "model_type"])
            if not cc.empty:
                color = _model_choice_color(cc["model_type"])
                fig, ax = plt.subplots()
                sc = ax.scatter(cc["center_folded_0_180"], cc[tilt_col], c=color, cmap="coolwarm", s=20)
                cbar = fig.colorbar(sc, ax=ax)
                cbar.set_ticks([-1, 0, 1])
                cbar.set_ticklabels(["other", "single", "two_plane"])
                ax.set_xlabel("center_true_folded_0_180")
                ax.set_ylabel("tilt_true")
                ax.set_title("Model choice map")
                fig.tight_layout()
                fig.savefig(global_root / "model_choice_map.png", dpi=150)
                plt.close(fig)

        if health_plots:
            _plot_status_counts(summary, global_root / "status_counts.png")
            _plot_hist(summary, "runtime_seconds", global_root / "runtime_hist.png", "Runtime distribution")

    return summary
