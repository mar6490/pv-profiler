"""General utility functions."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def setup_logging(level: str | int = logging.INFO) -> None:
    """Initialize application logging."""
    resolved = logging.getLevelName(level.upper()) if isinstance(level, str) else level
    logging.basicConfig(
        level=resolved,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def utc_timestamp_label() -> str:
    """Return deterministic run folder label."""
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
