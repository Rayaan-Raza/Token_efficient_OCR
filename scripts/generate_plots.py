#!/usr/bin/env python3
"""Generate plots from metric CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import outputs_path
from src.visualization.plot_budget_curves import plot_cer_vs_budget


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-csv", default=str(outputs_path("metrics", "ocr_metrics.csv")))
    args = parser.parse_args()
    csv_path = Path(args.ocr_csv)
    if csv_path.exists():
        plot_cer_vs_budget(csv_path, outputs_path("plots", "cer_vs_budget.png"))
        print("Wrote plot to outputs/plots/cer_vs_budget.png")
    else:
        print(f"No OCR metrics at {csv_path}")


if __name__ == "__main__":
    main()
