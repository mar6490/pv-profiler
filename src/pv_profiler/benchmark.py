from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def circular_err_deg(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    d = abs((float(a) - float(b)) % 360)
    return min(d, 360 - d)


def _pick(row: pd.Series, candidates: list[str]) -> Any:
    for c in candidates:
        if c in row.index and pd.notna(row[c]):
            return row[c]
    return None


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
        sid = str(row[system_id_col])
        out_dir = output_root / sid
        result_path = out_dir / "08_orientation_result.json"

        gt_tilt = _pick(row, ["tilt_true", "true_tilt", "tilt_deg", "tilt"])
        gt_az = _pick(row, ["azimuth_true", "true_azimuth", "azimuth_deg", "azimuth"])
        gt_center = _pick(row, ["center_true", "azimuth_center_true", "center_deg"])
        gt_weight = _pick(row, ["weight_true", "weight_east_true", "weight"])

        if not result_path.exists():
            rows.append({"system_id": sid, "status": "no_result"})
            continue

        res = json.loads(result_path.read_text(encoding="utf-8"))
        model_type = res.get("model_type")

        tilt_est = res.get("tilt_deg")
        az_est = res.get("azimuth_deg")
        center_est = res.get("azimuth_center_deg")
        weight_est = res.get("weight_east")

        rows.append(
            {
                "system_id": sid,
                "status": "ok",
                "model_type": model_type,
                "tilt_true": gt_tilt,
                "tilt_est": tilt_est,
                "tilt_abs_err": abs(float(tilt_est) - float(gt_tilt)) if tilt_est is not None and gt_tilt is not None else None,
                "azimuth_true": gt_az,
                "azimuth_est": az_est,
                "az_circular_err_deg": circular_err_deg(az_est, gt_az),
                "center_true": gt_center,
                "center_est": center_est,
                "center_circular_err_deg": circular_err_deg(center_est, gt_center),
                "weight_true": gt_weight,
                "weight_est": weight_est,
                "weight_abs_err": abs(float(weight_est) - float(gt_weight)) if weight_est is not None and gt_weight is not None else None,
                "score_rmse": res.get("score_rmse"),
                "score_bic": res.get("score_bic"),
                "timing_total_s": ((res.get("timing_seconds") or {}).get("total")),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_root / "benchmark_results.csv", index=False)
    return df
