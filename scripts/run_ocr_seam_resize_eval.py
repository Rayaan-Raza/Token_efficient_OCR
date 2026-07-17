#!/usr/bin/env python3
"""Driver for OCR-protected full-page compression + frozen RAVEN-Select.

Does not retune RAVEN-Select v1.0.0. Reuses cached BM25 / LER-BOPS metrics and
only swaps the full-page reader CSV.

Example::

    python scripts/run_ocr_seam_resize_eval.py --n 100 --variants margin_crop,ws_compress,ocr_seam
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.answer_selection.evaluate import write_select_summary
from src.answer_selection.method_spec import METHOD_VERSION, method_stamp
from src.answer_selection.ocr_presence import build_ocr_presence_cache
from src.answer_selection.train import evaluate_selector
from src.extraction.gates import RevGateResult, write_gate_report
from src.metrics.statistical_tests import bootstrap_ci
from src.preprocessing.ocr_page_compress import METHOD_LABELS
from src.utils.logging_utils import setup_experiment_logging
from src.utils.paths import outputs_path
import numpy as np
import pandas as pd


VARIANT_TO_METHOD = METHOD_LABELS  # margin_crop -> margin_crop_resize, ...


def _write_seam_gate(
    *,
    variant: str,
    n: int,
    seam_anls: float,
    resize_anls: float,
    orig_anls: float,
    vs_resize: dict,
    vs_orig: dict,
) -> RevGateResult:
    beats_resize = seam_anls > resize_anls and bool(vs_resize.get("ci_lower_positive"))
    beats_orig = seam_anls > orig_anls and bool(vs_orig.get("ci_lower_positive"))
    if beats_resize and beats_orig:
        status, passed = "PASS", True
    elif beats_resize:
        status, passed = "PARTIAL", False
    else:
        status, passed = "FAIL", False
    gate = RevGateResult(
        name="P26_ocr_seam_resize",
        passed=passed,
        metrics={
            "variant": variant,
            "n": n,
            "status": status,
            "method_version": METHOD_VERSION,
            "seam_anls": seam_anls,
            "resize_anls": resize_anls,
            "original_raven_anls": orig_anls,
            "vs_resize": vs_resize,
            "vs_original_raven": vs_orig,
        },
        thresholds={
            "ci_lower_vs_resize_gt_0": True,
            "ci_lower_vs_original_raven_gt_0": True,
        },
        message=(
            f"{status}: variant={variant} n={n} seam={seam_anls:.4f} "
            f"resize={resize_anls:.4f} orig={orig_anls:.4f}"
        ),
    )
    write_gate_report(f"P26_ocr_seam_resize_{variant}_n{n}", gate)
    return gate


def _swap_fullpage_csv(n: int, fullpage_method: str, dataset: str = "docvqa") -> str:
    """Copy full-page + BM25 + LER CSVs under a metrics tag for selector eval."""
    if fullpage_method == "resize":
        src = outputs_path("metrics", f"vlm_metrics_{dataset}_{n}_resize_single.csv")
    else:
        src = outputs_path(
            "metrics", f"vlm_metrics_{dataset}_{n}_{fullpage_method}_single.csv"
        )
    if not src.exists():
        raise FileNotFoundError(f"missing full-page metrics: {src}")

    tag = f"seam_{fullpage_method}"
    dst = outputs_path("metrics", f"vlm_metrics_{dataset}_{n}_resize_single_{tag}.csv")
    pd.read_csv(src).to_csv(dst, index=False)
    for fname in [
        f"vlm_metrics_{dataset}_{n}_bm25_only_k2.csv",
        f"vlm_metrics_{dataset}_{n}_learned_lgbm_strict_k2.csv",
    ]:
        base = outputs_path("metrics", fname)
        if not base.exists():
            raise FileNotFoundError(base)
        tagged = outputs_path("metrics", fname.replace(".csv", f"_{tag}.csv"))
        pd.read_csv(base).to_csv(tagged, index=False)
    return tag


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--dataset", default="docvqa")
    p.add_argument(
        "--variants",
        default="margin_crop,ws_compress,ocr_seam",
        help="Comma-separated compress variants",
    )
    p.add_argument("--write-gates", action="store_true")
    p.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip variants whose full-page VLM CSV is not ready",
    )
    args = p.parse_args()
    logger = setup_experiment_logging("ocr_seam_resize_eval")

    # Original RAVEN-Select baseline on frozen resize reader.
    orig = evaluate_selector(
        args.n, model_name="raven_select_rule", dataset=args.dataset
    )
    logger.info("Original RAVEN-Select ANLS=%.4f EM=%.4f", orig["anls"], orig["em"])

    summary_rows = []
    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        fullpage = VARIANT_TO_METHOD[variant]
        try:
            tag = _swap_fullpage_csv(args.n, fullpage, dataset=args.dataset)
        except FileNotFoundError as e:
            if args.skip_missing:
                logger.warning("skip %s: %s", variant, e)
                continue
            raise

        # Rebuild OCR presence under the tagged metrics so grounding uses new preds.
        build_ocr_presence_cache(
            args.n, metrics_tag=tag, dataset=args.dataset
        )
        result = evaluate_selector(
            args.n,
            model_name="raven_select_rule",
            metrics_tag=tag,
            dataset=args.dataset,
        )
        write_select_summary(result, tag=tag, dataset=args.dataset)

        # Paired vs original RAVEN-Select
        d_orig = bootstrap_ci(
            (np.array(result["anls_vec"]) - np.array(orig["anls_vec"])).tolist()
        )
        vs_orig = {
            "delta": d_orig[0],
            "ci95": [d_orig[1], d_orig[2]],
            "ci_lower_positive": d_orig[1] > 0,
        }
        vs_resize = result["vs_resize"]
        logger.info(
            "%s selector ANLS=%.4f EM=%.4f vs_resize_ci=%s vs_orig_ci=%s",
            variant,
            result["anls"],
            result["em"],
            vs_resize.get("ci95"),
            vs_orig.get("ci95"),
        )
        row = {
            "variant": variant,
            "fullpage_method": fullpage,
            "anls": result["anls"],
            "em": result["em"],
            "vs_resize": vs_resize,
            "vs_original_raven": vs_orig,
            "route_counts": result["route_counts"],
            "method": method_stamp(role="production", comparator_model="raven_select_rule"),
            "method_version": METHOD_VERSION,
            "metrics_tag": tag,
        }
        if args.write_gates:
            gate = _write_seam_gate(
                variant=variant,
                n=args.n,
                seam_anls=float(result["anls"]),
                resize_anls=float(result["baselines"]["resize"]["anls"]),
                orig_anls=float(orig["anls"]),
                vs_resize=vs_resize,
                vs_orig=vs_orig,
            )
            row["gate_status"] = gate.metrics["status"]
            row["gate_passed"] = gate.passed
        summary_rows.append(row)

    out = outputs_path("metrics", f"ocr_seam_resize_selector_n{args.n}.json")
    out.write_text(json.dumps({
        "n": args.n,
        "dataset": args.dataset,
        "original_raven_select": {"anls": orig["anls"], "em": orig["em"]},
        "variants": summary_rows,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(out), "n_variants": len(summary_rows)}, indent=2))


if __name__ == "__main__":
    main()
