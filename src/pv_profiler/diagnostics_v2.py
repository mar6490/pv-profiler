from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


KEY_FILES = [
    "00_status.json",
    "01_input_diagnostics.json",
    "02_sdt_daily_flags.csv",
    "05_power_fit.parquet",
    "07_p_norm_clear.parquet",
    "08_orientation_result.json",
]


def _parse_system_id(name: str) -> int | None:
    digits = "".join(ch for ch in name if ch.isdigit())
    if not digits:
        return None
    return int(digits)


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


def _plot_scatter(df: pd.DataFrame, x: str, y: str, out: Path, title: str) -> None:
    tmp = pd.DataFrame({x: pd.to_numeric(df[x], errors="coerce"), y: pd.to_numeric(df[y], errors="coerce")}).dropna()
    if tmp.empty:
        return
    fig, ax = plt.subplots()
    ax.scatter(tmp[x], tmp[y], s=14)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
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


def generate_diagnostics_v2(
    *,
    output_root: str | Path,
    systems_metadata_csv: str | Path | None = None,
    system_id_col: str = "system_id",
) -> pd.DataFrame:
    output_root = Path(output_root)
    diag_root = output_root / "diagnostics_v2"
    per_system_root = diag_root / "per_system"
    global_root = diag_root / "global"
    per_system_root.mkdir(parents=True, exist_ok=True)
    global_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for system_dir in sorted([p for p in output_root.iterdir() if p.is_dir()]):
        if system_dir.name == "diagnostics_v2":
            continue

        system_id = _parse_system_id(system_dir.name)
        status_path = system_dir / "00_status.json"
        orient_path = system_dir / "08_orientation_result.json"

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
        }
        for f in KEY_FILES:
            row[f"has_{f}"] = (system_dir / f).exists()
        rows.append(row)

        sys_out = per_system_root / system_dir.name
        sys_out.mkdir(parents=True, exist_ok=True)
        _plot_presence_bar({f: (system_dir / f).exists() for f in KEY_FILES}, sys_out / "artifact_presence.png")
        _plot_daily_flags(system_dir / "02_sdt_daily_flags.csv", sys_out / "daily_flags.png")

    summary = pd.DataFrame(rows)

    if systems_metadata_csv is not None and not summary.empty:
        md = pd.read_csv(systems_metadata_csv)
        if system_id_col in md.columns:
            md = md.copy()
            md[system_id_col] = pd.to_numeric(md[system_id_col], errors="coerce").astype("Int64")
            summary["system_id"] = pd.to_numeric(summary["system_id"], errors="coerce").astype("Int64")
            summary = summary.merge(md, left_on="system_id", right_on=system_id_col, how="left")

    summary.to_csv(diag_root / "aggregated_metrics.csv", index=False)

    if not summary.empty:
        _plot_status_counts(summary, global_root / "status_counts.png")
        _plot_hist(summary, "runtime_seconds", global_root / "runtime_hist.png", "Runtime distribution")
        _plot_hist(summary, "score_rmse", global_root / "rmse_hist.png", "RMSE distribution")
        _plot_scatter(summary, "tilt_deg", "azimuth_deg", global_root / "tilt_vs_azimuth.png", "Tilt vs azimuth")

    return summary
