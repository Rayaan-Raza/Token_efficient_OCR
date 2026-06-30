#!/usr/bin/env python3
"""Regenerate paper tables from metrics CSVs."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.utils.paths import outputs_path, repo_path


def main() -> None:
    paper_tables = repo_path("paper", "tables")
    paper_tables.mkdir(parents=True, exist_ok=True)
    ocr_csv = outputs_path("metrics", "ocr_metrics.csv")
    vlm_csv = outputs_path("metrics", "vlm_metrics.csv")
    if ocr_csv.exists():
        df = pd.read_csv(ocr_csv)
        if "invalid_budget" in df.columns:
            valid = df[df["invalid_budget"].fillna(False).astype(bool) == False]
        else:
            valid = df
        summary = valid.groupby(["method", "budget"]).agg(
            cer_mean=("cer", "mean"), wer_mean=("wer", "mean"), n=("cer", "count")
        ).reset_index()
        summary.to_csv(paper_tables / "table_ocr_budget.csv", index=False)
        print(f"Wrote {paper_tables / 'table_ocr_budget.csv'}")
    if vlm_csv.exists():
        df = pd.read_csv(vlm_csv)
        summary = df.groupby(["method", "num_patches"]).agg(
            em_mean=("exact_match", "mean"), anls_mean=("anls", "mean"), n=("anls", "count")
        ).reset_index()
        summary.to_csv(paper_tables / "table_vlm_patches.csv", index=False)
        print(f"Wrote {paper_tables / 'table_vlm_patches.csv'}")


if __name__ == "__main__":
    main()
