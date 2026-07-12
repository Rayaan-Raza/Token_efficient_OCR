#!/usr/bin/env python3
"""Evaluate answer_coverage@K for all fair selector methods."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.data.dataset_loader import iter_manifest
from src.metrics.answer_coverage import answer_in_selected_patches, compute_patch_labels, find_answer_boxes, patch_from_dict
from src.metrics.cost_accounting import CostRecord
from src.metrics.gates import load_headline_k
from src.preprocessing.selectors import FAIR_METHODS, select_patches
from src.utils.image_io import load_image
from src.utils.logging_utils import log_cost_breakdown, log_progress, log_section, setup_experiment_logging
from src.utils.ocr_cache import load_cached_ocr_boxes
from src.utils.paths import outputs_path


def _image_id(record: dict) -> str:
    return record.get("image_id") or record.get("doc_id") or record.get("question_id", "unknown")


def _answers(record: dict) -> list[str]:
    ans = record.get("answers") or record.get("answer") or []
    if isinstance(ans, str):
        return [ans]
    return list(ans)


def _patch_texts_for_selection(iid: str, selected_indices: set[int]) -> list[str]:
    path = outputs_path("ocr", "patches", f"{iid}.json")
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [p.get("text", "") for p in data["patches"] if p.get("index") in selected_indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch coverage@K by method.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--methods", default=",".join(sorted(FAIR_METHODS)))
    parser.add_argument("--k", default="")
    parser.add_argument("--random-seeds", default="0-9")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    logger = setup_experiment_logging("eval_coverage")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    hk = int(args.k) if args.k else load_headline_k(2)
    log_section(logger, f"Coverage eval | K={hk} | methods={methods}")

    if "-" in args.random_seeds:
        a, b = args.random_seeds.split("-")
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(x) for x in args.random_seeds.split(",")]

    records = list(iter_manifest(args.manifest))
    if args.limit:
        records = records[: args.limit]

    labels_df = None
    lp = outputs_path("labels", "patch_labels.parquet")
    if lp.exists():
        labels_df = pd.read_parquet(lp)

    rows = []
    costs: list[CostRecord] = []

    for method in methods:
        method_hits = []
        for rec in records:
            iid = _image_id(rec)
            q = rec.get("question", "")
            answers = _answers(rec)
            image = load_image(rec["image_path"])
            boxes = load_cached_ocr_boxes(iid) or []

            patch_labels = None
            if labels_df is not None and method == "oracle":
                sub = labels_df[(labels_df["question_id"] == rec.get("question_id")) & (labels_df["image_id"] == iid)]
                patch_labels = [r.to_dict() for _, r in sub.iterrows()]

            if method == "random":
                seed_hits = []
                for seed in seeds:
                    t0 = time.perf_counter()
                    sel = select_patches(image, method, hk, q, boxes, seed=seed, eval_labels=(method == "oracle"), patch_labels=patch_labels, answers=answers if method == "oracle" else None)
                    sel_idx = {p.index for p in sel.patches}
                    texts = _patch_texts_for_selection(iid, sel_idx)
                    seed_hits.append(answer_in_selected_patches(answers, texts))
                    costs.append(CostRecord(f"{rec.get('question_id')}_{seed}", method, selection_s=time.perf_counter() - t0, num_candidates=sel.meta["num_candidates"], num_patches_selected=len(sel.patches)))
                method_hits.append(sum(seed_hits) / len(seed_hits))
            else:
                t0 = time.perf_counter()
                sel = select_patches(image, method, hk, q, boxes, eval_labels=(method == "oracle"), patch_labels=patch_labels, answers=answers if method == "oracle" else None)
                sel_idx = {p.index for p in sel.patches}
                texts = _patch_texts_for_selection(iid, sel_idx)
                method_hits.append(float(answer_in_selected_patches(answers, texts)))
                costs.append(CostRecord(rec.get("question_id", ""), method, selection_s=time.perf_counter() - t0, num_candidates=sel.meta["num_candidates"], num_patches_selected=len(sel.patches)))

        cov = sum(method_hits) / len(method_hits) if method_hits else 0.0
        rows.append({"method": method, "k": hk, "coverage": cov, "n": len(method_hits)})
        logger.info("%s coverage@%d = %.3f", method, hk, cov)

    out = outputs_path("metrics", "coverage_by_method.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "k", "coverage", "n"])
        w.writeheader()
        w.writerows(rows)
    logger.info("Wrote %s", out)

    cost_out = outputs_path("metrics", "cost_by_method_coverage.csv")
    with open(cost_out, "w", encoding="utf-8", newline="") as f:
        if costs:
            w = csv.DictWriter(f, fieldnames=costs[0].as_dict().keys())
            w.writeheader()
            w.writerows([c.as_dict() for c in costs])
    logger.info("Wrote %s", cost_out)


if __name__ == "__main__":
    main()
