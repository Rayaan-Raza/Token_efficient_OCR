#!/usr/bin/env python3
"""Evaluate evidence_coverage@K and ocr_exact_coverage@K for all selectors."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.data.dataset_loader import iter_manifest
from src.metrics.answer_coverage import mean_rank_of_first_positive, rank_candidates_by_score
from src.metrics.coverage_eval import eval_selection
from src.metrics.cost_accounting import CostRecord
from src.metrics.gates import load_headline_k
from src.preprocessing.selectors import FAIR_METHODS, select_patches
from src.utils.image_io import load_image
from src.utils.logging_utils import log_section, setup_experiment_logging
from src.utils.ocr_cache import load_cached_ocr_boxes
from src.utils.paths import outputs_path


def _image_id(record: dict) -> str:
    return record.get("image_id") or record.get("doc_id") or record.get("question_id", "unknown")


def _answers(record: dict) -> list[str]:
    ans = record.get("answers") or record.get("answer") or []
    if isinstance(ans, str):
        return [ans]
    return list(ans)


def _labels_for_image(labels_df: pd.DataFrame, iid: str) -> list[dict]:
    sub = labels_df[labels_df["image_id"] == iid]
    return [r.to_dict() for _, r in sub.iterrows()]


def _patch_texts_for_selection(iid: str, selected_indices: set[int]) -> list[str]:
    path = outputs_path("ocr", "patches", f"{iid}.json")
    if not path.exists():
        return []
    import json
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [p.get("text", "") for p in data["patches"] if p.get("index") in selected_indices]


def _aggregate(samples: list[dict], key: str) -> float:
    if not samples:
        return 0.0
    first = samples[0].get(key)
    if isinstance(first, (int, float)) and not isinstance(first, bool):
        return sum(float(s[key]) for s in samples) / len(samples)
    return sum(1 for s in samples if s[key]) / len(samples)


def _mean_optional(values: list[float | None]) -> float:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _eval_row(
    sel,
    iid: str,
    rec: dict,
    method: str,
    hk: int,
    patch_labels: list[dict],
    labels_by_index: dict,
    answers: list[str],
) -> dict:
    sel_idx = {p.index for p in sel.patches}
    texts = _patch_texts_for_selection(iid, sel_idx)
    ranked = rank_candidates_by_score(sel.candidate_pool, sel.meta.get("candidate_scores", []))
    ev = eval_selection(patch_labels, sel_idx, answers, texts, ranked)
    row = {
        "image_id": iid,
        "question_id": rec.get("question_id", ""),
        "method": method,
        "k": hk,
        "evidence_strict": float(ev["evidence_strict"]),
        "evidence_soft": float(ev["evidence_soft"]),
        "evidence_any": float(ev["evidence_any"]),
        "evidence_coverage": float(ev["evidence_any"]),
        "ocr_exact_coverage": float(ev["ocr_exact_coverage"]),
        "box_overlap_coverage": float(ev["box_overlap_coverage"]),
        "soft_token_coverage": float(ev["soft_token_coverage"]),
        "fuzzy_coverage": float(ev["fuzzy_coverage"]),
        "mean_rank_of_first_positive": ev["mean_rank_first_positive_any"] or 0.0,
        "runtime": sel.meta.get("selection_s", 0.0),
        "num_candidates": sel.meta["num_candidates"],
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Evidence and OCR-exact coverage@K by method.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--methods", default=",".join(sorted(FAIR_METHODS | {"oracle"})))
    parser.add_argument("--k", default="", help="Headline K or comma-separated sweep e.g. 1,2,3,4,6,8")
    parser.add_argument("--output-tag", default="", help="Output suffix e.g. k4 -> coverage_by_method_k4.csv")
    parser.add_argument("--random-seeds", default="0-9")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    logger = setup_experiment_logging("eval_coverage")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if args.k and "," in args.k:
        k_values = [int(x.strip()) for x in args.k.split(",") if x.strip()]
    elif args.k:
        k_values = [int(args.k)]
    else:
        k_values = [load_headline_k(2)]
    log_section(logger, f"Coverage eval | K={k_values} | methods={methods}")

    if "-" in args.random_seeds:
        a, b = args.random_seeds.split("-")
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(x) for x in args.random_seeds.split(",")]

    records = list(iter_manifest(args.manifest))
    if args.limit:
        records = records[: args.limit]

    labels_path = outputs_path("labels", "patch_labels.parquet")
    if not labels_path.exists():
        logger.error("Missing %s", labels_path)
        sys.exit(1)
    labels_df = pd.read_parquet(labels_path)

    per_question_rows: list[dict] = []
    costs: list[CostRecord] = []
    summary_rows: list[dict] = []

    metric_keys = [
        "evidence_strict", "evidence_soft", "evidence_any", "evidence_coverage",
        "ocr_exact_coverage", "box_overlap_coverage", "soft_token_coverage", "fuzzy_coverage",
        "mean_rank_of_first_positive", "runtime", "num_candidates",
    ]

    for hk in k_values:
        for method in methods:
            samples: list[dict] = []

            for rec in records:
                iid = _image_id(rec)
                q = rec.get("question", "")
                answers = _answers(rec)
                image = load_image(rec["image_path"])
                boxes = load_cached_ocr_boxes(iid) or []
                patch_labels = _labels_for_image(labels_df, iid)
                labels_by_index = {
                    int(l.get("patch_index", l.get("index", -1))): l for l in patch_labels
                }

                if method == "random":
                    seed_hits = []
                    for seed in seeds:
                        t0 = time.perf_counter()
                        sel = select_patches(image, method, hk, q, boxes, seed=seed)
                        sel.meta["selection_s"] = time.perf_counter() - t0
                        row = _eval_row(sel, iid, rec, method, hk, patch_labels, labels_by_index, answers)
                        seed_hits.append(row)
                        costs.append(CostRecord(
                            f"{rec.get('question_id', iid)}_{seed}", method,
                            selection_s=row["runtime"],
                            num_candidates=row["num_candidates"],
                            num_patches_selected=len(sel.patches),
                        ))
                    merged = {
                        key: sum(float(h[key]) for h in seed_hits) / len(seed_hits)
                        for key in metric_keys
                    }
                    merged.update({
                        "image_id": iid,
                        "question_id": rec.get("question_id", ""),
                        "method": method,
                        "k": hk,
                    })
                    samples.append(merged)
                    per_question_rows.append(merged)
                else:
                    t0 = time.perf_counter()
                    sel = select_patches(
                        image, method, hk, q, boxes,
                        eval_labels=(method == "oracle"),
                        patch_labels=patch_labels if method == "oracle" else None,
                        answers=answers if method == "oracle" else None,
                    )
                    sel.meta["selection_s"] = time.perf_counter() - t0
                    row = _eval_row(sel, iid, rec, method, hk, patch_labels, labels_by_index, answers)
                    samples.append(row)
                    per_question_rows.append(row)
                    costs.append(CostRecord(
                        rec.get("question_id", iid), method,
                        selection_s=row["runtime"],
                        num_candidates=row["num_candidates"],
                        num_patches_selected=len(sel.patches),
                    ))

            summary = {"method": method, "k": hk, "n": len(samples)}
            for key in metric_keys:
                summary[key] = _aggregate(samples, key)
            summary_rows.append(summary)
            logger.info(
                "%s | K=%d strict=%.3f any=%.3f ocr_exact=%.3f",
                method, hk, summary["evidence_strict"], summary["evidence_any"],
                summary["ocr_exact_coverage"],
            )

    tag = args.output_tag or ("k_sweep" if len(k_values) > 1 else "")
    suffix = f"_{tag}" if tag else ""
    out_fields = ["method", "k"] + metric_keys + ["n"]
    out = outputs_path("metrics", f"coverage_by_method{suffix}.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(summary_rows)
    logger.info("Wrote %s", out)

    per_q = outputs_path("metrics", f"coverage_per_question{suffix}.csv")
    with open(per_q, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_question_rows[0].keys()))
        w.writeheader()
        w.writerows(per_question_rows)
    logger.info("Wrote %s (%d rows)", per_q, len(per_question_rows))

    cost_out = outputs_path("metrics", f"cost_by_method_coverage{suffix}.csv")
    with open(cost_out, "w", encoding="utf-8", newline="") as f:
        if costs:
            w = csv.DictWriter(f, fieldnames=costs[0].as_dict().keys())
            w.writeheader()
            w.writerows([c.as_dict() for c in costs])
    logger.info("Wrote %s", cost_out)


if __name__ == "__main__":
    main()
