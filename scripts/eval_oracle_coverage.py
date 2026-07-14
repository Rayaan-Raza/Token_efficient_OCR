#!/usr/bin/env python3
"""Compute oracle_ocr_exact@K and oracle_evidence@K ceilings."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.data.dataset_loader import iter_manifest
from src.metrics.answer_coverage import (
    answer_in_selected_patches,
    evidence_in_selected,
    oracle_select_patches,
    patch_from_dict,
    pool_reachability_rates,
)
from src.utils.logging_utils import log_section, setup_experiment_logging
from src.utils.paths import outputs_path


def _image_id(record: dict) -> str:
    return record.get("image_id") or record.get("doc_id") or record.get("question_id", "unknown")


def _answers(record: dict) -> list[str]:
    ans = record.get("answers") or record.get("answer") or []
    if isinstance(ans, str):
        return [ans]
    return list(ans)


def _patch_texts_for_indices(iid: str, selected_indices: set[int]) -> list[str]:
    path = outputs_path("ocr", "patches", f"{iid}.json")
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [p.get("text", "") for p in data["patches"] if p.get("index") in selected_indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle OCR-exact and evidence coverage@K.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--k", default="1,2,4,8")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    logger = setup_experiment_logging("oracle_coverage")
    ks = [int(x) for x in args.k.split(",")]
    log_section(logger, f"Oracle ceilings | K={ks}")

    labels_path = outputs_path("labels", "patch_labels.parquet")
    if not labels_path.exists():
        logger.error("Missing %s — run label_patches_from_answers.py first", labels_path)
        sys.exit(1)
    df = pd.read_parquet(labels_path)

    records = list(iter_manifest(args.manifest))
    if args.limit:
        records = records[: args.limit]

    ocr_hits: dict[int, list[bool]] = defaultdict(list)
    evidence_hits: dict[int, list[bool]] = defaultdict(list)
    pool_rows: list[dict[str, bool]] = []

    for rec in records:
        iid = _image_id(rec)
        answers = _answers(rec)
        sub = df[df["image_id"] == iid]
        if sub.empty:
            continue
        patches = [patch_from_dict(r) for _, r in sub.iterrows()]
        labels = [r.to_dict() for _, r in sub.iterrows()]
        pool_rows.append(pool_reachability_rates(labels))

        for k in ks:
            selected = oracle_select_patches(patches, labels, k)
            sel_idx = {p.index for p in selected}
            texts = _patch_texts_for_indices(iid, sel_idx)
            ocr_hits[k].append(answer_in_selected_patches(answers, texts))
            evidence_hits[k].append(evidence_in_selected(labels, sel_idx))

    rows = []
    for k in ks:
        n = len(ocr_hits[k])
        ocr_cov = sum(ocr_hits[k]) / n if n else 0.0
        ev_cov = sum(evidence_hits[k]) / n if n else 0.0
        rows.append({
            "k": k,
            "oracle_ocr_exact": ocr_cov,
            "oracle_evidence": ev_cov,
            "n": n,
        })
        logger.info("oracle_ocr_exact@%d = %.3f | oracle_evidence@%d = %.3f (n=%d)", k, ocr_cov, k, ev_cov, n)

    out = outputs_path("metrics", "oracle_coverage_by_k.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k", "oracle_ocr_exact", "oracle_evidence", "n"])
        w.writeheader()
        w.writerows(rows)
    logger.info("Wrote %s", out)

    if pool_rows:
        reach = {
            key: sum(1 for r in pool_rows if r[key]) / len(pool_rows)
            for key in pool_rows[0].keys()
        }
        reach_out = outputs_path("metrics", "candidate_reachability.csv")
        with open(reach_out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["metric", "rate", "n"])
            w.writeheader()
            for metric, rate in reach.items():
                w.writerow({"metric": metric, "rate": rate, "n": len(pool_rows)})
                logger.info("%s = %.3f", metric, rate)
        logger.info("Wrote %s", reach_out)


if __name__ == "__main__":
    main()
