#!/usr/bin/env python3
"""Categorize VLM QA failures from merged metrics CSV (Phase 15).

Reads ``outputs/metrics/vlm_metrics_merged.csv`` (or legacy single file),
extracts non-exact-match rows, and writes failure case tables.

Run::

    python scripts/analyze_failures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.utils.paths import outputs_path


def _load_vlm_df() -> pd.DataFrame | None:
    merged = outputs_path("metrics", "vlm_metrics_merged.csv")
    if merged.exists() and merged.stat().st_size > 0:
        return pd.read_csv(merged)
    legacy = outputs_path("metrics", "vlm_metrics.csv")
    return pd.read_csv(legacy) if legacy.exists() else None


def main() -> None:
    """Extract and summarize VLM failures."""
    df = _load_vlm_df()
    if df is None or len(df) == 0:
        print("No VLM metrics found")
        return

    pred_col = "parsed_prediction" if "parsed_prediction" in df.columns else "prediction"
    if "dry_run" in df.columns:
        df = df[df["dry_run"].fillna(False).astype(bool) == False]

    failures = df[df["exact_match"] < 1.0].copy()
    failures["failure_type"] = "other"
    failures.loc[failures[pred_col].astype(str).str.len() < 2, "failure_type"] = "empty_or_short"
    out = outputs_path("failure_cases", "vlm_failures.csv")
    failures.to_csv(out, index=False)
    summary = failures.groupby("failure_type").size().reset_index(name="count")
    summary.to_csv(outputs_path("failure_cases", "failure_summary.csv"), index=False)
    print(f"Wrote {out} ({len(failures)} failures)")


if __name__ == "__main__":
    main()
