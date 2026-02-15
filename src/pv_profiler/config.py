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
    "pipeline": {
        "fit_tau": 0.03,
        "clipping_threshold_day_share": 0.10,
        "clipping_fraction_day_median": 0.01,
        "clear_time_fraction_min": 0.005,
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
