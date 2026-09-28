"""Logging setup shared by all front ends."""

import logging

from rich.logging import RichHandler


def setup_logging(level="info"):
    """Install a single rich handler on the root logger; it shows level and file:line, so the format is just the message."""
    level_number = logging.getLevelName(str(level).upper())
    if not isinstance(level_number, int):
        raise ValueError(f"Unknown log level {level!r}; use one of debug, info, warning, error, critical")
    logging.basicConfig(level=level_number, format="%(message)s", handlers=[RichHandler(show_time=False)], force=True)
