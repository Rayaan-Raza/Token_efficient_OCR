"""Logging setup for CLI scripts and long-running QE-BOPS experiments."""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.paths import outputs_path


def setup_logging(
    name: str = "bops",
    log_file: Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create or retrieve a configured logger."""
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


def new_run_id(stage: str) -> str:
    """Generate a timestamped run id for logs and gate reports."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stage}_{ts}"


def setup_experiment_logging(stage: str, run_id: str | None = None) -> logging.Logger:
    """Configure console + file logging under outputs/logs/."""
    rid = run_id or new_run_id(stage)
    log_file = outputs_path("logs", f"{rid}.log")
    logger = setup_logging(name=f"bops.{stage}", log_file=log_file)
    logger.info("=== QE-BOPS experiment stage=%s run_id=%s ===", stage, rid)
    logger.info("Log file: %s", log_file)
    return logger


def log_section(logger: logging.Logger, title: str) -> None:
    """Print a banner section header."""
    bar = "=" * 60
    logger.info(bar)
    logger.info(title)
    logger.info(bar)


def log_progress(logger: logging.Logger, i: int, n: int, detail: str = "") -> None:
    """Log progress line i/n with optional detail."""
    msg = f"[{i}/{n}]"
    if detail:
        msg = f"{msg} {detail}"
    logger.info(msg)


def log_gate_result(
    logger: logging.Logger,
    gate_name: str,
    passed: bool,
    metrics: dict[str, Any],
    threshold: str,
    *,
    level: str = "min",
) -> None:
    """Log gate PASS/FAIL with metrics."""
    status = "PASS" if passed else "FAIL"
    logger.info("GATE %s (%s): %s", gate_name, level, status)
    logger.info("  threshold: %s", threshold)
    for k, v in metrics.items():
        logger.info("  %s = %s", k, v)


def log_cost_breakdown(logger: logging.Logger, sample_id: str, timings: dict[str, float]) -> None:
    """Log per-sample cost breakdown."""
    parts = ", ".join(f"{k}={v:.3f}s" for k, v in sorted(timings.items()))
    logger.info("COST sample=%s | %s", sample_id, parts)
