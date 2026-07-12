#!/usr/bin/env python3
"""Plot coverage@K vs ANLS transfer curve."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.logging_utils import setup_experiment_logging
from src.utils.paths import outputs_path, repo_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Coverage-ANLS transfer plot.")
    args = parser.parse_args()
    logger = setup_experiment_logging("transfer_plot")

    cov_path = outputs_path("metrics", "coverage_by_method.csv")
    vlm_path = outputs_path("metrics", "vlm_metrics_merged.csv")
    if not cov_path.exists():
        logger.error("Missing %s", cov_path)
        sys.exit(1)

    coverage = {r["method"]: float(r["coverage"]) for r in csv.DictReader(open(cov_path, encoding="utf-8"))}
    anls_by_method: dict[str, list[float]] = {}
    if vlm_path.exists():
        for r in csv.DictReader(open(vlm_path, encoding="utf-8")):
            m = r.get("method", "")
            if r.get("anls"):
                anls_by_method.setdefault(m, []).append(float(r["anls"]))

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib required")
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(6, 4))
    for method, cov in coverage.items():
        if method not in anls_by_method:
            continue
        mean_anls = sum(anls_by_method[method]) / len(anls_by_method[method])
        ax.scatter(cov, mean_anls, label=method)
        ax.annotate(method, (cov, mean_anls), fontsize=7)

    ax.set_xlabel("answer_coverage@K")
    ax.set_ylabel("ANLS")
    ax.set_title("Coverage → VLM transfer")
    ax.legend(fontsize=6, loc="best")
    out = repo_path("paper", "figures", "coverage_anls_transfer.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
