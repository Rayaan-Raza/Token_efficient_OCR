#!/usr/bin/env python3
"""Bar chart of evidence reachability ceilings for the paper."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.logging_utils import setup_experiment_logging
from src.utils.paths import outputs_path, repo_path


def _read_rate(path: Path, metric: str) -> float | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("metric") == metric:
                return float(row["rate"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot evidence reachability ceilings.")
    args = parser.parse_args()
    logger = setup_experiment_logging("reachability_plot")

    gap_path = outputs_path("metrics", "coverage_gap_summary.csv")
    reach_path = outputs_path("metrics", "candidate_reachability.csv")

    fullpage = _read_rate(gap_path, "fullpage_answer_present")
    ocr_exact = _read_rate(reach_path, "candidate_ocr_exact_reachability")
    evidence = _read_rate(reach_path, "candidate_evidence_reachability")

    if fullpage is None:
        with open(gap_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("metric") == "fullpage_answer_present":
                    fullpage = float(row["rate"])
                    break
    if ocr_exact is None:
        ocr_exact = _read_rate(gap_path, "patch_ocr_exact_positive")
    if evidence is None:
        evidence = _read_rate(gap_path, "any_candidate_positive")

    if None in (fullpage, ocr_exact, evidence):
        logger.error("Missing reachability summary inputs; run diagnose_coverage_gap.py and eval_oracle_coverage.py")
        sys.exit(1)

    labels = [
        "Full-page OCR\nanswer present",
        "Candidate OCR-exact\nreachability",
        "Candidate evidence\nreachability",
    ]
    values = [fullpage * 100, ocr_exact * 100, evidence * 100]
    colors = ["#59A14F", "#E15759", "#4C78A8"]

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib required")
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=colors)
    for bar, val in zip(bars, values):
        ax.annotate(f"{val:.0f}%", (bar.get_x() + bar.get_width() / 2, val),
                    ha="center", va="bottom", fontsize=10)

    ax.set_ylim(0, 100)
    ax.set_ylabel("% questions with reachable evidence")
    ax.set_title("Candidate-pool reachability under different evidence definitions")
    ax.grid(axis="y", alpha=0.3)

    caption = (
        "Candidate-pool reachability under different evidence definitions. "
        "Exact patch OCR underestimates visual evidence availability because cropped "
        "patches sometimes lose the exact OCR string even when the full-page OCR box "
        "or fuzzy evidence remains available."
    )
    fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=8, wrap=True)

    for out_dir in (repo_path("paper", "figures"), repo_path("paper", "latex_v2", "figures")):
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "evidence_reachability_ceilings.pdf"
        fig.savefig(out, bbox_inches="tight")
        logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
