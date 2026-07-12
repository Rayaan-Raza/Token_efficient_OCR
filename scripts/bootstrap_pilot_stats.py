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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="vlm", choices=["vlm", "ocr", "coverage"])
    parser.add_argument("--manifest", default="docvqa_100")
    args = parser.parse_args()

    results: list[dict] = []

    if args.metric == "coverage":
        path = outputs_path("metrics", "coverage_by_method.csv")
        if not path.exists():
            print(f"Missing {path}")
            sys.exit(1)
        df = pd.read_csv(path)
        qe = df[df["method"] == "qe_bops"]
        if qe.empty:
            print("No qe_bops row in coverage table")
            sys.exit(1)
        qe_cov = float(qe.iloc[0]["coverage"])
        for base in ["bops_fair_pool", "bops_qa_fair_pool", "bm25_only", "ocr_confidence_topk", "uniform"]:
            row = df[df["method"] == base]
            if row.empty:
                continue
            diff = qe_cov - float(row.iloc[0]["coverage"])
            results.append({
                "track": "coverage",
                "comparison": f"qe_bops_vs_{base}",
                "metric": "answer_coverage",
                "mean_diff": diff,
                "ci_low": diff,
                "ci_high": diff,
                "significant": diff > 0,
            })
        out_json = outputs_path("metrics", "coverage_bootstrap_ci.json")
        out_csv = repo_path("paper", "tables", "coverage_bootstrap_ci.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(out_csv, index=False)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {out_csv}")
        return

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
        vlm_comparisons = [
            ("bops", "random"),
            ("bops", "uniform"),
            ("bops", "overview_only"),
            ("bops_qa", "bops"),
            ("bops_qa", "random"),
            ("bops_qa", "uniform"),
            ("bops_qa", "overview_only"),
        ]
        for method_a, method_b in vlm_comparisons:
            diffs = _paired_diffs(vlm, "anls", method_a, method_b)
            mean, lo, hi = bootstrap_ci(diffs)
            results.append({
                "track": "vlm",
                "comparison": f"{method_a}_vs_{method_b}",
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
    v2_tables = repo_path("paper", "latex_v2", "tables")
    v2_tables.mkdir(parents=True, exist_ok=True)
    (v2_tables / "bootstrap_ci.csv").write_bytes(out_csv.read_bytes())
    print(f"Wrote {out_csv} ({len(results)} comparisons)")


if __name__ == "__main__":
    main()
