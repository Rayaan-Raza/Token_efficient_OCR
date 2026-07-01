#!/usr/bin/env python3
"""Orchestrate multi-phase OCR/VLM experiments and ablations.

Use ``--dry-run`` for pipeline validation or ``--real`` for actual inference.
Child scripts inherit the mode explicitly — dry-run is never hardcoded.

Run::

    python scripts/run_full_experiment.py --phase debug --dry-run
    python scripts/run_full_experiment.py --phase pilot --real
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    """Execute a subprocess command in the repo root."""
    print(">", " ".join(cmd))
    subprocess.check_call(cmd, cwd=REPO_ROOT)


def _ocr_cmd(
    py: str,
    manifest: str,
    methods: list[str],
    budgets: list[str],
    limit: int,
    dry_run: bool,
    stage: str,
) -> list[str]:
    cmd = [
        py, "scripts/run_ocr_eval.py",
        "--manifest", manifest,
        "--methods", *methods,
        "--budgets", *budgets,
        "--limit", str(limit),
        "--experiment-stage", stage,
        "--overwrite",
    ]
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def _vlm_cmd(
    py: str,
    manifest: str,
    method: str,
    num_patches: int,
    limit: int,
    dry_run: bool,
    stage: str,
) -> list[str]:
    cmd = [
        py, "scripts/run_vlm_eval.py",
        "--manifest", manifest,
        "--method", method,
        "--num-patches", str(num_patches),
        "--limit", str(limit),
        "--experiment-stage", stage,
        "--overwrite",
    ]
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def main() -> None:
    """Dispatch experiment phase."""
    parser = argparse.ArgumentParser(description="Run BOPS experiment phases.")
    parser.add_argument("--phase", default="pilot", choices=["debug", "pilot", "paper", "ablation"])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Use placeholder OCR/VLM outputs")
    mode.add_argument("--real", action="store_true", help="Run real OCR/VLM inference")
    args = parser.parse_args()

    dry_run = args.dry_run or not args.real
    banner = "DRY-RUN" if dry_run else "REAL"
    print(f"MODE: {banner} | phase={args.phase}")
    py = sys.executable

    OCR_BUDGETS = ["area_1.0", "area_0.5", "area_0.25", "area_0.125", "kb_500", "kb_200", "kb_100", "kb_50"]
    BOPS_BUDGETS = ["patches_2", "patches_4", "patches_8", "patches_12"]

    if args.phase == "debug":
        run(_ocr_cmd(py, "data/manifests/textocr_debug.jsonl",
                     ["original", "resize"], ["area_0.5"], 5, dry_run, "debug"))
        run(_vlm_cmd(py, "data/manifests/docvqa_debug.jsonl", "resize", 4, 2, dry_run, "debug"))
    elif args.phase == "pilot":
        run(_ocr_cmd(py, "data/manifests/textocr_pilot.jsonl",
                     ["original", "resize", "jpeg", "webp"], OCR_BUDGETS, 20, dry_run, "pilot"))
    elif args.phase == "ablation":
        for k in (0, 2, 4, 8, 12):
            run(_vlm_cmd(py, "data/manifests/docvqa_pilot.jsonl", "bops", k, 10, dry_run, "pilot"))
        run(_ocr_cmd(py, "data/manifests/textocr_debug.jsonl",
                     ["bops"], ["patches_4"], 5, dry_run, "pilot"))
    elif args.phase == "paper":
        run(_ocr_cmd(py, "data/manifests/textocr_pilot.jsonl",
                     ["original", "resize", "jpeg", "webp", "bops"],
                     OCR_BUDGETS + BOPS_BUDGETS, 50, dry_run, "paper"))
        for method in ("resize", "overview_only", "random", "uniform", "bops"):
            run(_vlm_cmd(py, "data/manifests/docvqa_pilot.jsonl", method, 2, 10, dry_run, "paper"))
        run([py, "scripts/merge_vlm_metrics.py"])
        run([py, "scripts/analyze_failures.py"])
        run([py, "scripts/make_paper_assets.py"])

    run([py, "scripts/generate_plots.py"])


if __name__ == "__main__":
    main()
