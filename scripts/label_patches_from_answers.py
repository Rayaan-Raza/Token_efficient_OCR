#!/usr/bin/env python3
"""Build hierarchical patch labels from GT answers (eval/training only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.data.dataset_loader import iter_manifest
from src.metrics.answer_coverage import compute_patch_labels, find_answer_boxes, patch_from_dict
from src.utils.logging_utils import log_progress, log_section, setup_experiment_logging
from src.utils.ocr_cache import load_cached_ocr_boxes
from src.utils.paths import outputs_path


def _image_id(record: dict) -> str:
    return record.get("image_id") or record.get("doc_id") or record.get("question_id", "unknown")


def _answers(record: dict) -> list[str]:
    ans = record.get("answers") or record.get("answer") or []
    if isinstance(ans, str):
        return [ans]
    return list(ans)


def main() -> None:
    parser = argparse.ArgumentParser(description="Label patches from GT answers.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    logger = setup_experiment_logging("label_patches")
    log_section(logger, "Hierarchical patch labeling (GT eval only)")

    records = list(iter_manifest(args.manifest))
    if args.limit:
        records = records[: args.limit]

    rows = []
    for i, rec in enumerate(records):
        iid = _image_id(rec)
        qid = rec.get("question_id", "")
        log_progress(logger, i + 1, len(records), qid)
        answers = _answers(rec)

        patch_path = outputs_path("ocr", "patches", f"{iid}.json")
        if not patch_path.exists():
            continue
        with open(patch_path, encoding="utf-8") as f:
            patch_data = json.load(f)

        full_boxes = load_cached_ocr_boxes(iid) or []
        answer_boxes = find_answer_boxes(full_boxes, answers)

        for p in patch_data["patches"]:
            patch = patch_from_dict(p)
            labels = compute_patch_labels(p.get("text", ""), patch, answers, answer_boxes)
            rows.append({
                "image_id": iid,
                "question_id": qid,
                "question": rec.get("question", ""),
                "patch_index": p.get("index", 0),
                **patch.as_dict(),
                **labels,
            })

    out = outputs_path("labels", "patch_labels.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    logger.info("Wrote %s (%d rows)", out, len(rows))


if __name__ == "__main__":
    main()
