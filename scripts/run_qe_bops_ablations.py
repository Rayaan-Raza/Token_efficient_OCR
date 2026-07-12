#!/usr/bin/env python3
"""Run QE-BOPS ablation variants."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.logging_utils import setup_experiment_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="QE-BOPS ablations.")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    logger = setup_experiment_logging("ablations")
    methods = "qe_bops,qe_bops_no_question,bops_fair_pool,bops_qa_fair_pool,bm25_only"
    cmd = [
        sys.executable, "scripts/eval_patch_coverage.py",
        "--manifest", str(args.manifest),
        "--methods", methods,
    ]
    logger.info("RUN %s", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    main()
