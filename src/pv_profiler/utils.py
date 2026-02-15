"""General utility functions."""

import logging


def setup_logging(level: int = logging.INFO) -> None:
    """Initialize application logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
