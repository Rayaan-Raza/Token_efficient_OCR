#!/usr/bin/env python3
"""Run heuristic QE-BOPS selection and log scores (wrapper for coverage eval)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Heuristic QE-BOPS run.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval_patch_coverage.py"),
        "--manifest", str(args.manifest),
        "--methods", "qe_bops,bops_fair_pool,bops_qa_fair_pool",
        "--k", str(args.k),
    ]
    if args.verbose:
        print("RUN", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    main()
