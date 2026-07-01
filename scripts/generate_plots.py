#!/usr/bin/env python3
"""Generate budget degradation plots from OCR metric CSVs (Phase 5, 16).

Reads ``outputs/metrics/ocr_metrics.csv`` and writes CER-vs-budget curves to
``outputs/plots/``. Excludes invalid-budget rows.

Run::

    python scripts/generate_plots.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import outputs_path
from src.visualization.plot_budget_curves import plot_cer_vs_budget


def main() -> None:
    """CLI: plot CER vs budget if metrics CSV exists."""
    parser = argparse.ArgumentParser(description="Generate experiment plots.")
    parser.add_argument("--ocr-csv", default=None)
    args = parser.parse_args()
    if args.ocr_csv:
        csv_path = Path(args.ocr_csv)
    else:
        metrics_dir = outputs_path("metrics")
        candidates = sorted(metrics_dir.glob("ocr_metrics*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        csv_path = candidates[0] if candidates else metrics_dir / "ocr_metrics.csv"
    if csv_path.exists() and csv_path.stat().st_size > 0:
        plot_cer_vs_budget(csv_path, outputs_path("plots", "cer_vs_budget.png"))
        print(f"Wrote plot from {csv_path.name} to outputs/plots/cer_vs_budget.png")
    else:
        print(f"No OCR metrics at {csv_path}")


if __name__ == "__main__":
    main()
