#!/usr/bin/env python3
"""Orchestrate multi-phase OCR/VLM experiments and ablations.

Subcommands via ``--phase``:
    - debug: tiny smoke run (5 OCR, 2 VLM rows)
    - pilot: OCR curves on 20 TextOCR samples
    - ablation: patch-count and scoring sweeps
    - paper: larger OCR + multi-method VLM + failure analysis + tables

Always ends by regenerating plots from ``ocr_metrics.csv``.

Run::

    python scripts/run_full_experiment.py --phase pilot
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    """Execute a subprocess command in the repo root.

    Args:
        cmd: argv list (e.g. ``[python, script, ...]``).

    Raises:
        subprocess.CalledProcessError: On non-zero exit.
    """
    print(">", " ".join(cmd))
    subprocess.check_call(cmd, cwd=REPO_ROOT)


def main() -> None:
    """Dispatch experiment phase and regenerate plots."""
    parser = argparse.ArgumentParser(description="Run BOPS experiment phases.")
    parser.add_argument("--phase", default="pilot", choices=["debug", "pilot", "paper", "ablation"])
    args = parser.parse_args()
    py = sys.executable

    if args.phase == "debug":
        run([py, "scripts/run_ocr_eval.py", "--manifest", "data/manifests/textocr_debug.jsonl",
             "--methods", "original", "resize", "--budgets", "area_0.5", "--limit", "5", "--dry-run"])
        run([py, "scripts/run_vlm_eval.py", "--manifest", "data/manifests/docvqa_debug.jsonl",
             "--limit", "2", "--dry-run"])
    elif args.phase == "pilot":
        run([py, "scripts/run_ocr_eval.py", "--manifest", "data/manifests/textocr_pilot.jsonl",
             "--methods", "original", "resize", "jpeg", "webp", "bops",
             "--budgets", "area_0.5", "area_0.25", "kb_200", "--limit", "20", "--dry-run"])
    elif args.phase == "ablation":
        for k in (0, 2, 4, 8, 12):
            run([py, "scripts/run_vlm_eval.py", "--manifest", "data/manifests/docvqa_pilot.jsonl",
                 "--method", "bops", "--num-patches", str(k), "--limit", "10", "--dry-run"])
        for _ in range(3):
            run([py, "scripts/run_ocr_eval.py", "--manifest", "data/manifests/textocr_debug.jsonl",
                 "--methods", "bops", "--budgets", "patches_4", "--limit", "5", "--dry-run"])
    elif args.phase == "paper":
        run([py, "scripts/run_ocr_eval.py", "--manifest", "data/manifests/textocr_pilot.jsonl",
             "--methods", "original", "resize", "jpeg", "webp", "bops",
             "--budgets", "area_1.0", "area_0.5", "area_0.25", "kb_500", "kb_200", "--limit", "50", "--dry-run"])
        for method in ("resize", "overview_only", "random", "uniform", "bops"):
            run([py, "scripts/run_vlm_eval.py", "--manifest", "data/manifests/docvqa_pilot.jsonl",
                 "--method", method, "--num-patches", "4", "--limit", "10", "--dry-run"])
        run([py, "scripts/analyze_failures.py"])
        run([py, "scripts/make_paper_assets.py"])
    run([py, "scripts/generate_plots.py"])


if __name__ == "__main__":
    main()
