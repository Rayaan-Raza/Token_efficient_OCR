#!/usr/bin/env python3
"""Evaluate QE-BOPS automated gates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.metrics.gates import GATE_FUNCS, write_gate_report
from src.utils.logging_utils import log_gate_result, log_section, setup_experiment_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Check QE-BOPS gates.")
    parser.add_argument("--gate", action="append", dest="gates", help="Gate name e.g. G1_candidates")
    parser.add_argument("--all", action="store_true", help="Run all implemented gates")
    args = parser.parse_args()

    logger = setup_experiment_logging("check_gates")
    names = list(GATE_FUNCS.keys()) if args.all else (args.gates or ["G1_candidates"])
    log_section(logger, f"Checking gates: {', '.join(names)}")

    results = []
    for name in names:
        if name not in GATE_FUNCS:
            logger.error("Unknown gate: %s", name)
            sys.exit(1)
        res = GATE_FUNCS[name]()
        log_gate_result(logger, res.name, res.passed, res.metrics, res.threshold, level=res.level)
        if res.message:
            logger.info("  note: %s", res.message)
        results.append(res)

    report = write_gate_report(results)
    logger.info("Gate report: %s", report)
    sys.exit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    main()
