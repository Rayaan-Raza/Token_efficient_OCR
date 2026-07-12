#!/usr/bin/env python3
"""TextOCR auxiliary sanity track for shared overview+patch infra."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.logging_utils import setup_experiment_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="TextOCR auxiliary OCR eval.")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    logger = setup_experiment_logging("textocr_aux")
    cmd = [
        sys.executable, "scripts/run_ocr_eval.py",
        "--manifest", "data/manifests/textocr_pilot.jsonl",
        "--methods", "bops", "resize",
        "--budgets", "patches_8", "area_0.25",
        "--limit", str(args.limit),
        "--experiment-stage", "pilot",
    ]
    logger.info("RUN %s", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    main()
