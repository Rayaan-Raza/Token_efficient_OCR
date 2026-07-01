#!/usr/bin/env python3
"""Merge per-method VLM metric CSVs into a single comparison table.

Globs ``outputs/metrics/vlm_metrics_*.csv`` (excluding ``*_merged.csv``),
concatenates rows, deduplicates, and writes ``vlm_metrics_merged.csv``.

Run::

    python scripts/merge_vlm_metrics.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.utils.paths import outputs_path


def merge_vlm_metrics(metrics_dir: Path | None = None) -> Path:
    """Merge per-method VLM CSV files into one dataframe and save.

    Args:
        metrics_dir: Directory containing ``vlm_metrics_*.csv`` files.

    Returns:
        Path to ``vlm_metrics_merged.csv``.
    """
    metrics_dir = metrics_dir or outputs_path("metrics")
    pattern_files = sorted(metrics_dir.glob("vlm_metrics_*.csv"))
    files = [f for f in pattern_files if "_merged" not in f.name]

    if not files:
        print("No vlm_metrics_*.csv files found.")
        out = metrics_dir / "vlm_metrics_merged.csv"
        out.write_text("", encoding="utf-8")
        return out

    frames = []
    for f in files:
        df = pd.read_csv(f)
        print(f"  {f.name}: {len(df)} rows")
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    dedupe_cols = [c for c in ("run_id", "image_id", "method", "num_patches") if c in merged.columns]
    if dedupe_cols:
        before = len(merged)
        merged = merged.drop_duplicates(subset=dedupe_cols, keep="last")
        if len(merged) < before:
            print(f"  Deduped {before - len(merged)} duplicate rows")

    out = metrics_dir / "vlm_metrics_merged.csv"
    merged.to_csv(out, index=False)
    print(f"Wrote {len(merged)} rows to {out}")
    return out


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Merge VLM metric CSV files.")
    parser.add_argument("--metrics-dir", default=None, help="Override metrics directory")
    args = parser.parse_args()
    metrics_dir = Path(args.metrics_dir) if args.metrics_dir else None
    merge_vlm_metrics(metrics_dir)


if __name__ == "__main__":
    main()
