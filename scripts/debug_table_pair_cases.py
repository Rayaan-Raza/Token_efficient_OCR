#!/usr/bin/env python3
"""Debug hard table/label-value cases for qe_bops_table_pair."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset_loader import iter_manifest
from src.preprocessing.patch_scoring_qa import patch_ocr_text
from src.preprocessing.selectors import select_patches
from src.utils.image_io import load_image
from src.utils.logging_utils import log_section, setup_experiment_logging
from src.utils.ocr_cache import load_cached_ocr_boxes
from src.utils.paths import outputs_path

DEFAULT_CASES = [
    "docvqa_val_57403",
    "docvqa_val_57415",
    "docvqa_val_32871",
    "docvqa_val_32879",
    "docvqa_val_49177",
]


def _image_id(record: dict) -> str:
    return record.get("image_id") or record.get("doc_id", "")


def _answers(record: dict) -> list[str]:
    ans = record.get("answers") or record.get("answer") or []
    if isinstance(ans, str):
        return [ans]
    return list(ans)


def _sel_text(iid: str, patches) -> str:
    boxes = load_cached_ocr_boxes(iid) or []
    return " || ".join(patch_ocr_text(p, boxes) for p in patches)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug table-pair hard cases.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES))
    parser.add_argument("--margin", type=float, default=0.05)
    args = parser.parse_args()

    logger = setup_experiment_logging("debug_table_pair")
    log_section(logger, "Table-pair hard case debug")

    case_ids = {c.strip() for c in args.cases.split(",") if c.strip()}
    records = { _image_id(r): r for r in iter_manifest(args.manifest) if _image_id(r) in case_ids }

    out_dir = outputs_path("debug", "table_pair_cases")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for iid in sorted(case_ids):
        rec = records.get(iid)
        if not rec:
            logger.warning("Missing manifest row for %s", iid)
            continue
        image = load_image(rec["image_path"])
        boxes = load_cached_ocr_boxes(iid) or []
        q = rec.get("question", "")

        sel_q = select_patches(image, "bops_qa_fair_pool", 2, q, boxes)
        sel_np = select_patches(image, "qe_bops_node_pair", 2, q, boxes)
        sel_tp = select_patches(
            image, "qe_bops_table_pair", 2, q, boxes,
        )

        row = {
            "image_id": iid,
            "question": q,
            "answers": " | ".join(_answers(rec)),
            "q_bops_ocr": _sel_text(iid, sel_q.patches),
            "node_pair_ocr": _sel_text(iid, sel_np.patches),
            "table_pair_ocr": _sel_text(iid, sel_tp.patches),
            "matched_row_text": sel_tp.meta.get("matched_row_text", ""),
            "selected_label": sel_tp.meta.get("best_label_text", ""),
            "selected_value": sel_tp.meta.get("best_value_text", ""),
            "relation_type": sel_tp.meta.get("best_relation") or sel_tp.meta.get("cross_patch_relation", ""),
            "boost_applied": sel_tp.meta.get("cooccurrence_boost", 0.0),
            "slot2_source": sel_tp.meta.get("slot2_source", ""),
            "slot2_swap": sel_tp.meta.get("slot2_swap", False),
            "table_pair_score": sel_tp.meta.get("table_pair_score", ""),
        }
        rows.append(row)
        logger.info("%s | slot2=%s | rel=%s | boost=%s", iid, row["slot2_source"], row["relation_type"], row["boost_applied"])

    csv_path = out_dir / "hard_cases.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)

    with open(out_dir / "hard_cases.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    logger.info("Wrote %s (%d cases)", csv_path, len(rows))


if __name__ == "__main__":
    main()
