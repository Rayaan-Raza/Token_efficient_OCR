#!/usr/bin/env python3
"""Regenerate paper tables from experiment metric CSVs (Phase 16).

Filters rows for paper eligibility:
    - dry_run == false
    - invalid_budget == false
    - not_applicable == false
    - experiment_stage in (pilot, paper)

Run::

    python scripts/make_paper_assets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.utils.experiment_io import filter_paper_dataframe
from src.utils.paths import outputs_path, repo_path


def _load_ocr_metrics() -> pd.DataFrame | None:
    """Load newest OCR metrics CSV (manifest-specific or legacy)."""
    metrics_dir = outputs_path("metrics")
    candidates = sorted(metrics_dir.glob("ocr_metrics*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    for c in candidates:
        if c.name != "ocr_metrics_merged.csv":
            return pd.read_csv(c)
    legacy = metrics_dir / "ocr_metrics.csv"
    return pd.read_csv(legacy) if legacy.exists() else None


def _load_vlm_metrics() -> pd.DataFrame | None:
    """Load merged VLM metrics, or legacy single file."""
    merged = outputs_path("metrics", "vlm_metrics_merged.csv")
    if merged.exists() and merged.stat().st_size > 0:
        return pd.read_csv(merged)
    legacy = outputs_path("metrics", "vlm_metrics.csv")
    return pd.read_csv(legacy) if legacy.exists() else None


def main() -> None:
    """Build paper table CSVs from filtered experiment outputs."""
    paper_tables = repo_path("paper", "tables")
    paper_tables.mkdir(parents=True, exist_ok=True)

    ocr_df = _load_ocr_metrics()
    if ocr_df is not None and len(ocr_df):
        valid = filter_paper_dataframe(ocr_df)
        print(f"OCR: {len(valid)} / {len(ocr_df)} rows eligible for paper tables")
        if len(valid):
            summary = valid.groupby(["method", "budget"]).agg(
                cer_mean=("cer", "mean"),
                wer_mean=("wer", "mean"),
                word_recall_mean=("word_recall", "mean"),
                n=("cer", "count"),
            ).reset_index()
            summary.to_csv(paper_tables / "table_ocr_budget.csv", index=False)
            print(f"Wrote {paper_tables / 'table_ocr_budget.csv'}")

    vlm_df = _load_vlm_metrics()
    if vlm_df is not None and len(vlm_df):
        valid = filter_paper_dataframe(vlm_df)
        print(f"VLM: {len(valid)} / {len(vlm_df)} rows eligible for paper tables")
        if len(valid):
            summary = valid.groupby(["method", "num_patches"]).agg(
                em_mean=("exact_match", "mean"),
                anls_mean=("anls", "mean"),
                n=("anls", "count"),
            ).reset_index()
            summary.to_csv(paper_tables / "table_vlm_patches.csv", index=False)
            print(f"Wrote {paper_tables / 'table_vlm_patches.csv'}")


if __name__ == "__main__":
    main()
