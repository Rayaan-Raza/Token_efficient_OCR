#!/usr/bin/env python3
"""Compute oracle answer coverage@K (diagnostic upper bound)."""

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
from src.metrics.answer_coverage import answer_in_selected_patches, oracle_select_patches, patch_from_dict
from src.utils.logging_utils import log_section, setup_experiment_logging
from src.utils.paths import outputs_path


def _image_id(record: dict) -> str:
    return record.get("image_id") or record.get("doc_id") or record.get("question_id", "unknown")


def _answers(record: dict) -> list[str]:
    ans = record.get("answers") or record.get("answer") or []
    if isinstance(ans, str):
        return [ans]
    return list(ans)


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle coverage@K analysis.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--k", default="1,2,4,8")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    logger = setup_experiment_logging("oracle_coverage")
    ks = [int(x) for x in args.k.split(",")]
    log_section(logger, f"Oracle coverage | K={ks}")

    labels_path = outputs_path("labels", "patch_labels.parquet")
    if not labels_path.exists():
        logger.error("Missing %s — run label_patches_from_answers.py first", labels_path)
        sys.exit(1)
    df = pd.read_parquet(labels_path)

    records = list(iter_manifest(args.manifest))
    if args.limit:
        records = records[: args.limit]

    hits: dict[int, list[bool]] = defaultdict(list)

    for rec in records:
        qid = rec.get("question_id", "")
        iid = _image_id(rec)
        answers = _answers(rec)
        sub = df[(df["question_id"] == qid) & (df["image_id"] == iid)]
        if sub.empty:
            continue
        patches = [patch_from_dict(r) for _, r in sub.iterrows()]
        labels = [r.to_dict() for _, r in sub.iterrows()]
        for k in ks:
            selected = oracle_select_patches(patches, labels, k)
            sel_idx = {p.index for p in selected}
            texts = [str(r["text"]) if "text" in r else "" for _, r in sub.iterrows() if r["patch_index"] in sel_idx or r["index"] in sel_idx]
            # fallback: use patch OCR from labels join
            texts = []
            for _, r in sub.iterrows():
                p = patch_from_dict(r)
                if p.index in sel_idx or int(r.get("patch_index", -1)) in sel_idx:
                    patch_ocr_path = outputs_path("ocr", "patches", f"{iid}.json")
                    if patch_ocr_path.exists():
                        with open(patch_ocr_path, encoding="utf-8") as f:
                            pdata = json.load(f)
                        for po in pdata["patches"]:
                            if po.get("index") == p.index:
                                texts.append(po.get("text", ""))
            hits[k].append(answer_in_selected_patches(answers, texts))

    rows = []
    for k in ks:
        cov = sum(hits[k]) / len(hits[k]) if hits[k] else 0.0
        rows.append({"k": k, "oracle_coverage": cov, "n": len(hits[k])})
        logger.info("oracle@%d = %.3f (n=%d)", k, cov, len(hits[k]))

    out = outputs_path("metrics", "oracle_coverage_by_k.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k", "oracle_coverage", "n"])
        w.writeheader()
        w.writerows(rows)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
