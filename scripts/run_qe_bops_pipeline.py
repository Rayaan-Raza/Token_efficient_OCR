#!/usr/bin/env python3
"""QE-BOPS master pipeline orchestrator with human approval stops."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.logging_utils import log_section, setup_experiment_logging
from src.utils.paths import data_path


PHASES = {
    "G0": [],
    "G1": [
        ["python", "scripts/build_docvqa_manifests.py"],
        ["python", "scripts/run_fullpage_ocr.py", "--manifest", "MANIFEST_100"],
        ["python", "scripts/generate_candidate_patches.py", "--manifest", "MANIFEST_100"],
        ["python", "scripts/check_gates.py", "--gate", "G1_candidates"],
    ],
    "G2": [
        ["python", "scripts/run_patch_ocr.py", "--manifest", "MANIFEST_100"],
        ["python", "scripts/label_patches_from_answers.py", "--manifest", "MANIFEST_100"],
        ["python", "scripts/eval_oracle_coverage.py", "--manifest", "MANIFEST_100"],
        ["python", "scripts/check_gates.py", "--gate", "G2_oracle"],
    ],
    "G3": [
        ["python", "scripts/diagnose_coverage_gap.py", "--manifest", "MANIFEST_100"],
        ["python", "scripts/plot_evidence_reachability.py"],
        ["python", "scripts/eval_patch_coverage.py", "--manifest", "MANIFEST_100"],
        ["python", "scripts/bootstrap_pilot_stats.py", "--metric", "coverage"],
        ["python", "scripts/check_gates.py", "--gate", "G3_heuristic"],
    ],
    "G4": [
        ["python", "scripts/run_vlm_eval.py", "--manifest", "MANIFEST_100", "--method", "qe_bops"],
    ],
    "G5": [
        ["python", "scripts/run_vlm_eval.py", "--manifest", "MANIFEST_300", "--method", "qe_bops"],
    ],
}


def _resolve(cmd: list[str], manifest_100: Path, manifest_300: Path) -> list[str]:
    out = []
    for part in cmd:
        if part == "MANIFEST_100":
            out.append(str(manifest_100))
        elif part == "MANIFEST_300":
            out.append(str(manifest_300))
        else:
            out.append(part)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QE-BOPS pipeline phases.")
    parser.add_argument("--stage", default="G1", help="Phase: G0,G1,G2,G3,G4,G5 or through_G3")
    parser.add_argument("--manifest-100", type=Path, default=data_path("manifests", "docvqa_100.jsonl"))
    parser.add_argument("--manifest-300", type=Path, default=data_path("manifests", "docvqa_300.jsonl"))
    parser.add_argument("--continue-after-approval", action="store_true")
    parser.add_argument("--no-approval-stop", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logger = setup_experiment_logging("pipeline")
    if args.no_approval_stop:
        logger.warning("--no-approval-stop: auto-advancing without human approval checkpoint")

    stages = []
    if args.stage.startswith("through"):
        target = args.stage.split("_", 1)[-1].upper()
        order = ["G0", "G1", "G2", "G3", "G4", "G5"]
        stages = order[: order.index(target) + 1] if target in order else [args.stage.upper()]
    else:
        stages = [args.stage.upper()]

    for stage in stages:
        log_section(logger, f"PHASE {stage}")
        cmds = PHASES.get(stage, [])
        if stage == "G0":
            cmds = [["python", "-m", "pytest", "tests/", "-q"], ["python", "scripts/check_gates.py", "--gate", "G0_baseline"]]
        for cmd in cmds:
            resolved = _resolve(cmd, args.manifest_100, args.manifest_300)
            logger.info("RUN: %s", " ".join(resolved))
            rc = subprocess.call(resolved, cwd=str(REPO_ROOT))
            if rc != 0:
                logger.error("Phase %s failed (exit %d). Stopping.", stage, rc)
                sys.exit(rc)
        if not args.no_approval_stop:
            logger.info("=== PHASE %s COMPLETE — awaiting approval before next phase ===", stage)
            if not args.continue_after_approval:
                sys.exit(0)

    logger.info("Pipeline stage(s) complete.")


if __name__ == "__main__":
    main()
