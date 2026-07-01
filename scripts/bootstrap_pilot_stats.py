#!/usr/bin/env python3
"""Bootstrap confidence intervals on pilot OCR/VLM metrics.

Uses paired per-sample differences and :func:`src.metrics.statistical_tests.bootstrap_ci`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.metrics.statistical_tests import bootstrap_ci
from src.utils.experiment_io import filter_paper_dataframe
from src.utils.paths import outputs_path, repo_path


def _load_pilot_ocr() -> pd.DataFrame:
    metrics_dir = outputs_path("metrics")
    candidates = sorted(
        metrics_dir.glob("ocr_metrics*pilot*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No pilot OCR metrics CSV found")
    return filter_paper_dataframe(pd.read_csv(candidates[0]))


def _load_pilot_vlm() -> pd.DataFrame:
    path = outputs_path("metrics", "vlm_metrics_merged.csv")
    if not path.exists():
        raise FileNotFoundError("vlm_metrics_merged.csv not found")
    return filter_paper_dataframe(pd.read_csv(path))


def _paired_diffs(
    df: pd.DataFrame,
    metric: str,
    method_a: str,
    method_b: str,
    budget_a: str | None = None,
    budget_b: str | None = None,
) -> list[float]:
    """Paired differences method_a - method_b on shared image_ids."""
    a = df[df["method"] == method_a]
    b = df[df["method"] == method_b]
    if budget_a is not None:
        a = a[a["budget"] == budget_a]
    if budget_b is not None:
        b = b[b["budget"] == budget_b]
    merged = a.merge(b, on="image_id", suffixes=("_a", "_b"))
    if len(merged) == 0:
        return []
    return (merged[f"{metric}_a"] - merged[f"{metric}_b"]).tolist()


def main() -> None:
    """Compute bootstrap CIs and write paper table."""
    results: list[dict] = []

    try:
        ocr = _load_pilot_ocr()
        comparisons = [
            ("bops_p8_vs_resize_a025", "bops", "resize", "patches_8", "area_0.25"),
            ("bops_p8_vs_original", "bops", "original", "patches_8", "area_1.0"),
        ]
        for label, ma, mb, ba, bb in comparisons:
            diffs = _paired_diffs(ocr, "word_recall", ma, mb, ba, bb)
            mean, lo, hi = bootstrap_ci(diffs)
            results.append({
                "track": "ocr",
                "comparison": label,
                "metric": "word_recall",
                "mean_diff": mean,
                "ci_lower": lo,
                "ci_upper": hi,
                "n_pairs": len(diffs),
            })
    except FileNotFoundError as e:
        print(f"OCR skip: {e}")

    try:
        vlm = _load_pilot_vlm()
        for other in ("random", "uniform", "overview_only"):
            diffs = _paired_diffs(vlm, "anls", "bops", other)
            mean, lo, hi = bootstrap_ci(diffs)
            results.append({
                "track": "vlm",
                "comparison": f"bops_vs_{other}",
                "metric": "anls",
                "mean_diff": mean,
                "ci_lower": lo,
                "ci_upper": hi,
                "n_pairs": len(diffs),
            })
    except FileNotFoundError as e:
        print(f"VLM skip: {e}")

    out_csv = repo_path("paper", "tables", "bootstrap_ci.csv")
    out_json = outputs_path("metrics", "bootstrap_ci.json")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out_csv} ({len(results)} comparisons)")


if __name__ == "__main__":
    main()
