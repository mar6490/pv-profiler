"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "logging": {"level": "INFO"},
    "paths": {
        "output_root": "outputs",
        "plants_csv": "data/processed/plants.csv",
    },
    "input": {
        "timestamp_col": "timestamp",
        "power_col": "ac_power",
    },
    "pipeline": {
        "fit_tau": 0.03,
        "clipping_threshold_day_share": 0.10,
        "clipping_fraction_day_median": 0.01,
        "skip_clipping": True,
        "fix_shifts": True,
        "solver": "CLARABEL",
    },
    "orientation": {
        "az_min": 90,
        "az_max": 270,
        "coarse_tilt_step": 5,
        "coarse_az_step": 10,
        "refine_tilt_step": 1,
        "refine_az_step": 2,
        "transposition_model": "perez",
        "loss_mode": "median_daily_rmse",
        "enable_two_plane": True,
        "ew_improve_threshold": 0.10,
        "top_n": 50,
    },
    "capacity": {
        "poa_threshold_wm2": 600.0,
    },
    "shading": {
        "r_max": 1.2,
        "az_bin_deg": 5,
        "el_bin_deg": 5,
        "morning_sector_deg": [60, 150],
        "evening_sector_deg": [210, 300],
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML configuration from a file path."""
    import yaml

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping.")
    return _deep_merge(DEFAULT_CONFIG, data)
