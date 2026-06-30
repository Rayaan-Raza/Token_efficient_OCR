#!/usr/bin/env python3
"""Categorize VLM failures from results CSV."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.utils.paths import outputs_path


def main() -> None:
    vlm_csv = outputs_path("metrics", "vlm_metrics.csv")
    if not vlm_csv.exists():
        print("No VLM metrics found")
        return
    df = pd.read_csv(vlm_csv)
    failures = df[df["exact_match"] < 1.0].copy()
    failures["failure_type"] = "other"
    failures.loc[failures["prediction"].str.len() < 2, "failure_type"] = "empty_or_short"
    out = outputs_path("failure_cases", "vlm_failures.csv")
    failures.to_csv(out, index=False)
    summary = failures.groupby("failure_type").size().reset_index(name="count")
    summary.to_csv(outputs_path("failure_cases", "failure_summary.csv"), index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
