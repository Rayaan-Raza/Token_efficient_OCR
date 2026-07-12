#!/usr/bin/env python3
"""Optional MLP ranker (bonus ablation)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.logging_utils import setup_experiment_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLP ranker (optional).")
    args = parser.parse_args()
    logger = setup_experiment_logging("train_mlp")
    logger.info("Optional MLP ranker — use train_logreg_ranker.py for mandatory diagnostic.")
    try:
        import torch
    except ImportError:
        logger.warning("torch not installed; skipping MLP")
        sys.exit(0)
    logger.info("MLP training stub complete — extend when LGBM beats heuristic.")


if __name__ == "__main__":
    main()
