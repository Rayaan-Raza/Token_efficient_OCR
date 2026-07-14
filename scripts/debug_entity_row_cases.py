#!/usr/bin/env python3
"""Debug entity-row hard cases with hit/miss labels."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.data.dataset_loader import iter_manifest
from src.metrics.coverage_eval import eval_selection
from src.preprocessing.patch_scoring_qa import patch_ocr_text
from src.preprocessing.selectors import select_patches
from src.utils.image_io import load_image
from src.utils.logging_utils import log_section, setup_experiment_logging
from src.utils.ocr_cache import load_cached_ocr_boxes
from src.utils.paths import outputs_path

DEFAULT_CASES = [
    "docvqa_val_32871",
    "docvqa_val_57403",
    "docvqa_val_57415",
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


def _patch_texts(iid: str, indices: set[int]) -> list[str]:
    path = outputs_path("ocr", "patches", f"{iid}.json")
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [p.get("text", "") for p in data["patches"] if p.get("index") in indices]


def _sel_ocr(boxes, patches) -> str:
    return " || ".join(patch_ocr_text(p, boxes) for p in patches)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug entity-row hard cases.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES))
    args = parser.parse_args()

    logger = setup_experiment_logging("debug_entity_row")
    log_section(logger, "Entity-row hard case debug")

    case_ids = {c.strip() for c in args.cases.split(",") if c.strip()}
    records = {_image_id(r): r for r in iter_manifest(args.manifest) if _image_id(r) in case_ids}
    labels_df = pd.read_parquet(outputs_path("labels", "patch_labels.parquet"))

    out_dir = outputs_path("debug", "entity_row_cases")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for iid in sorted(case_ids):
        rec = records.get(iid)
        if not rec:
            continue
        image = load_image(rec["image_path"])
        boxes = load_cached_ocr_boxes(iid) or []
        q = rec.get("question", "")
        answers = _answers(rec)
        labels = [r.to_dict() for _, r in labels_df[labels_df["image_id"] == iid].iterrows()]

        sel_q = select_patches(image, "bops_qa_fair_pool", 2, q, boxes)
        sel_er = select_patches(image, "qe_bops_entity_row", 2, q, boxes)

        idx_q = {p.index for p in sel_q.patches}
        idx_er = {p.index for p in sel_er.patches}
        ev_q = eval_selection(labels, idx_q, answers, _patch_texts(iid, idx_q))
        ev_er = eval_selection(labels, idx_er, answers, _patch_texts(iid, idx_er))

        m = sel_er.meta
        rows.append({
            "image_id": iid,
            "question": q,
            "answer": " | ".join(answers),
            "entity_tokens": ",".join(m.get("entity_tokens", [])) if isinstance(m.get("entity_tokens"), list) else m.get("entity_tokens", ""),
            "field_tokens": ",".join(m.get("field_tokens", [])) if isinstance(m.get("field_tokens"), list) else m.get("field_tokens", ""),
            "answer_type": m.get("answer_type_hint", ""),
            "selected_row_text": m.get("selected_row_text", ""),
            "selected_value_text": m.get("selected_value_text", ""),
            "relation_type": m.get("relation_type", ""),
            "boost_applied": m.get("boost_applied", 0.0),
            "q_bops_ocr": _sel_ocr(boxes, sel_q.patches),
            "entity_row_ocr": _sel_ocr(boxes, sel_er.patches),
            "q_bops_strict": ev_q["evidence_strict"],
            "q_bops_any": ev_q["evidence_any"],
            "entity_row_strict": ev_er["evidence_strict"],
            "entity_row_any": ev_er["evidence_any"],
            "slot2_source": m.get("slot2_source", ""),
            "slot2_swap": m.get("slot2_swap", False),
        })
        logger.info(
            "%s | entity=%s | er strict=%s any=%s | q strict=%s any=%s",
            iid, rows[-1]["entity_tokens"],
            ev_er["evidence_strict"], ev_er["evidence_any"],
            ev_q["evidence_strict"], ev_q["evidence_any"],
        )

    csv_path = out_dir / "hard_cases.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)
    with open(out_dir / "hard_cases.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    logger.info("Wrote %s", csv_path)


if __name__ == "__main__":
    main()
