from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_float_or_none(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, str) and x.strip() == "":
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return None


def parse_two_azimuths(s: Any) -> tuple[float | None, float | None]:
    if s is None:
        return None, None
    if isinstance(s, str):
        t = s.strip()
        if t == "":
            return None, None
        if "/" in t:
            a, b = t.split("/", 1)
            return parse_float_or_none(a), parse_float_or_none(b)
    one = parse_float_or_none(s)
    if one is not None:
        return one, None
    return None, None


def circular_err_deg(a: float | None, b: float | None) -> float | None:
    fa = parse_float_or_none(a)
    fb = parse_float_or_none(b)
    if fa is None or fb is None:
        return np.nan
    d = abs((fa - fb) % 360)
    return float(min(d, 360 - d))


def _pick(row: pd.Series, candidates: list[str]) -> Any:
    for c in candidates:
        if c in row.index and pd.notna(row[c]):
            return row[c]
    return None


def _norm_system_type(x: Any) -> str:
    s = str(x).strip().lower() if x is not None and not pd.isna(x) else ""
    if "east-west" in s or "east_west" in s or "eastwest" in s:
        return "east-west"
    if s == "single":
        return "single"
    return s if s else "unknown"


def build_benchmark_results(
    output_root: str | Path,
    systems_metadata_csv: str | Path,
    *,
    system_id_col: str = "system_id",
) -> pd.DataFrame:
    output_root = Path(output_root)
    meta = pd.read_csv(systems_metadata_csv)

    rows: list[dict[str, Any]] = []
    for _, row in meta.iterrows():
        sid = int(row[system_id_col])
        out_dir = output_root / f"system_{sid:03d}"
        result_path = out_dir / "08_orientation_result.json"

        system_type = _norm_system_type(_pick(row, ["system_type"]))

        # Ground truth selection by system type
        if system_type == "single":
            gt_tilt = parse_float_or_none(_pick(row, ["tilt_deg_true", "tilt_true", "tilt"]))
            gt_az = parse_float_or_none(_pick(row, ["azimuth_deg_true", "azimuth_true", "true_azimuth", "azimuth_deg", "azimuth"]))
            gt_center = None
            gt_east = None
            gt_west = None
        else:
            gt_tilt = parse_float_or_none(_pick(row, ["tilt", "tilt_deg_true", "tilt_true"]))
            gt_center = parse_float_or_none(_pick(row, ["azimuth_center_deg_true", "center_true", "azimuth_center_true", "center_deg"]))
            gt_east = parse_float_or_none(_pick(row, ["azimuth_east_deg_true"]))
            gt_west = parse_float_or_none(_pick(row, ["azimuth_west_deg_true"]))
            if gt_east is None or gt_west is None:
                az_a, az_b = parse_two_azimuths(_pick(row, ["azimuth"]))
                gt_east = gt_east if gt_east is not None else az_a
                gt_west = gt_west if gt_west is not None else az_b
            gt_az = parse_float_or_none(_pick(row, ["azimuth_deg_true", "azimuth_true", "true_azimuth", "azimuth_deg"]))


        if not result_path.exists():
            rows.append({"system_id": sid, "system_type": system_type, "status": "no_result"})
            continue

        res = json.loads(result_path.read_text(encoding="utf-8"))
        model_type = res.get("model_type")

        tilt_est = parse_float_or_none(res.get("tilt_deg"))
        az_est = parse_float_or_none(res.get("azimuth_deg"))
        center_est = parse_float_or_none(res.get("azimuth_center_deg"))
        est_east = parse_float_or_none(res.get("azimuth_east_deg"))
        est_west = parse_float_or_none(res.get("azimuth_west_deg"))

        tilt_err = abs(tilt_est - gt_tilt) if tilt_est is not None and gt_tilt is not None else np.nan

        # single/equivalent az error
        az_err = circular_err_deg(az_est, gt_az)

        # east-west specific errors
        center_err = circular_err_deg(center_est, gt_center)

        plane_err = np.nan
        if None not in (est_east, est_west, gt_east, gt_west):
            e1 = circular_err_deg(est_east, gt_east) + circular_err_deg(est_west, gt_west)
            e2 = circular_err_deg(est_east, gt_west) + circular_err_deg(est_west, gt_east)
            plane_err = float(min(e1, e2) / 2.0)


        rows.append(
            {
                "system_id": sid,
                "system_type": system_type,
                "status": "ok",
                "model_type": model_type,
                "tilt_true": gt_tilt,
                "tilt_est": tilt_est,
                "tilt_abs_err": tilt_err,
                "tilt_abs_err_deg": tilt_err,
                "azimuth_true": gt_az,
                "azimuth_est": az_est,
                "az_circular_err_deg": az_err,
                "az_abs_err_deg": az_err,
                "center_true": gt_center,
                "center_est": center_est,
                "center_circular_err_deg": center_err,
                "az_center_abs_err_deg": center_err,
                "azimuth_east_true": gt_east,
                "azimuth_west_true": gt_west,
                "azimuth_east_est": est_east,
                "azimuth_west_est": est_west,
                "az_plane_abs_err_deg": plane_err,
                "score_rmse": parse_float_or_none(res.get("score_rmse")),
                "timing_total_s": parse_float_or_none(((res.get("timing_seconds") or {}).get("total"))),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_root / "benchmark_results.csv", index=False)
    return df
