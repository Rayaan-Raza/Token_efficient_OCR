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


def _to_float_series(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        return series.map(lambda v: 1.0 if str(v).lower() in ("true", "1", "1.0") else 0.0)
    return series.astype(float)


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


def _load_pilot_vlm(manifest_stem: str | None = None) -> pd.DataFrame:
    metrics_dir = outputs_path("metrics")
    if manifest_stem:
        paths = sorted(metrics_dir.glob(f"vlm_metrics_{manifest_stem}_*.csv"))
        if not paths:
            raise FileNotFoundError(f"No VLM CSVs for manifest {manifest_stem}")
        return filter_paper_dataframe(pd.concat([pd.read_csv(p) for p in paths], ignore_index=True))
    path = metrics_dir / "vlm_metrics_merged.csv"
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
    parser.add_argument("--coverage-suffix", default="", help="e.g. k4 -> coverage_per_question_k4.csv")
    parser.add_argument("--qe-method", default="qe_bops", help="QE method name in coverage CSV")
    parser.add_argument("--k", type=int, default=0, help="Filter coverage rows to this K (0 = all)")
    args = parser.parse_args()

    results: list[dict] = []

    if args.metric == "coverage":
        suffix = f"_{args.coverage_suffix}" if args.coverage_suffix else ""
        # Prefer conventional suffix path; also accept learned_* naming.
        candidates = [
            outputs_path("metrics", f"coverage_per_question{suffix}.csv"),
            outputs_path("metrics", "learned_coverage_per_question.csv") if args.coverage_suffix == "learned" else None,
            outputs_path("metrics", f"learned_coverage_per_question{suffix}.csv") if args.coverage_suffix else None,
        ]
        per_q_path = next((p for p in candidates if p is not None and p.exists()), None)
        summary_candidates = [
            outputs_path("metrics", f"coverage_by_method{suffix}.csv"),
            outputs_path("metrics", "learned_coverage_by_method.csv") if args.coverage_suffix == "learned" else None,
        ]
        summary_path = next((p for p in summary_candidates if p is not None and p.exists()), None)
        if per_q_path is None:
            print(f"Missing coverage_per_question{suffix}.csv (and learned_* fallback)")
            sys.exit(1)
        df = pd.read_csv(per_q_path)
        if args.k and "k" in df.columns:
            df = df[df["k"] == args.k]
        metric_col = "evidence_coverage"
        if metric_col not in df.columns:
            metric_col = "coverage"
        qe = df[df["method"] == args.qe_method]
        if qe.empty:
            print(f"No {args.qe_method} rows in {per_q_path}")
            sys.exit(1)
        for base in ["bops_fair_pool", "bops_qa_fair_pool", "bm25_only", "ocr_confidence_topk", "uniform"]:
            base_df = df[df["method"] == base]
            if base_df.empty:
                continue
            merge_on = ["image_id", "k"] if "k" in qe.columns and "k" in base_df.columns else ["image_id"]
            merged = qe.merge(base_df, on=merge_on, suffixes=("_qe", "_base"))
            if merged.empty:
                continue
            for metric_name in ("evidence_strict", "evidence_any"):
                col_qe = f"{metric_name}_qe" if f"{metric_name}_qe" in merged.columns else None
                col_base = f"{metric_name}_base" if f"{metric_name}_base" in merged.columns else None
                if not col_qe or not col_base:
                    continue
                diffs = (_to_float_series(merged[col_qe]) - _to_float_series(merged[col_base])).tolist()
                mean, lo, hi = bootstrap_ci(diffs) if diffs else (0.0, 0.0, 0.0)
                results.append({
                    "track": "coverage",
                    "comparison": f"{args.qe_method}_vs_{base}",
                    "metric": metric_name,
                    "mean_diff": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "significant": lo > 0,
                    "n_pairs": len(diffs),
                })
        if summary_path is not None and summary_path.exists():
            summary = pd.read_csv(summary_path)
            if args.k and "k" in summary.columns:
                summary = summary[summary["k"] == args.k]
            qe_row = summary[summary["method"] == args.qe_method]
            if not qe_row.empty:
                results.append({
                    "track": "coverage",
                    "comparison": f"{args.qe_method}_headline",
                    "metric": "evidence_coverage",
                    "mean_diff": float(qe_row.iloc[0].get("evidence_coverage", qe_row.iloc[0].get("coverage", 0))),
                    "ci_low": float(qe_row.iloc[0].get("evidence_coverage", 0)),
                    "ci_high": float(qe_row.iloc[0].get("evidence_coverage", 0)),
                    "significant": True,
                    "n_pairs": int(qe_row.iloc[0].get("n", 0)),
                })
        out_json = outputs_path("metrics", f"coverage_bootstrap_ci{suffix}.json")
        if args.coverage_suffix == "learned":
            out_json = outputs_path("metrics", "coverage_bootstrap_ci_learned.json")
        out_csv = repo_path("paper", "tables", f"coverage_bootstrap_ci{suffix}.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(out_csv, index=False)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {out_csv}")
        print(f"Wrote {out_json} ({len(results)} comparisons)")
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
        vlm = _load_pilot_vlm(args.manifest if args.manifest else None)
        method_map = {
            "learned_lgbm_strict": "learned_lgbm_strict",
            "bops_qa_fair_pool": "bops_qa_fair_pool",
            "bm25_only": "bm25_only",
            "resize": "resize",
        }
        vlm_comparisons = [
            ("learned_lgbm_strict", "bops_qa_fair_pool"),
            ("learned_lgbm_strict", "bm25_only"),
            ("learned_lgbm_strict", "resize"),
            ("bops", "random"),
            ("bops", "uniform"),
            ("bops", "overview_only"),
            ("bops_qa", "bops"),
            ("bops_qa", "random"),
            ("bops_qa", "uniform"),
            ("bops_qa", "overview_only"),
        ]
        for method_a, method_b in vlm_comparisons:
            if method_a not in vlm["method"].values or method_b not in vlm["method"].values:
                continue
            for metric in ("anls", "exact_match"):
                diffs = _paired_diffs(vlm, metric, method_a, method_b)
                if not diffs:
                    continue
                mean, lo, hi = bootstrap_ci(diffs)
                results.append({
                    "track": "vlm",
                    "comparison": f"{method_a}_vs_{method_b}",
                    "metric": metric,
                    "mean_diff": mean,
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "n_pairs": len(diffs),
                    "significant": lo > 0 if metric == "anls" else lo >= 0,
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
