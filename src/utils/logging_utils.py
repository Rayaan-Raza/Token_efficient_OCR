"""Logging setup for CLI scripts and long-running experiments.

Configures a named logger with console output and optional file logging.
Calling ``setup_logging`` twice with the same name returns the existing logger
without duplicating handlers.
"""

import logging
import sys
from pathlib import Path


def setup_logging(
    name: str = "bops",
    log_file: Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create or retrieve a configured logger.

    Args:
        name: Logger name (default ``"bops"``).
        log_file: If set, also append logs to this file.
        level: Logging level (default ``INFO``).

    Returns:
        Configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
