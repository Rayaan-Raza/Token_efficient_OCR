#!/usr/bin/env python3
"""Decompose full-page vs candidate-oracle answer reachability gaps."""

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
from src.metrics.answer_coverage import (
    answer_in_selected_patches,
    answer_in_text,
    find_answer_boxes,
    oracle_select_patches,
    patch_from_dict,
)
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


def _fullpage_text(boxes: list[dict]) -> str:
    return " ".join(b.get("text", "") for b in boxes)


def _load_patch_ocr_map(image_id: str) -> dict[int, str]:
    path = outputs_path("ocr", "patches", f"{image_id}.json")
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return {int(p.get("index", i)): p.get("text", "") for i, p in enumerate(payload.get("patches", []))}


def _oracle_patch_ocr_hit(
    sub: pd.DataFrame,
    image_id: str,
    answers: list[str],
    k: int,
) -> bool:
    patches = [patch_from_dict(r) for _, r in sub.iterrows()]
    labels = [r.to_dict() for _, r in sub.iterrows()]
    selected = oracle_select_patches(patches, labels, k)
    sel_idx = {p.index for p in selected}
    ocr_map = _load_patch_ocr_map(image_id)
    texts = [ocr_map.get(idx, "") for idx in sel_idx if idx in ocr_map]
    return answer_in_selected_patches(answers, texts)


def _oracle_label_hit(sub: pd.DataFrame, k: int) -> bool:
    patches = [patch_from_dict(r) for _, r in sub.iterrows()]
    labels = [r.to_dict() for _, r in sub.iterrows()]
    selected = oracle_select_patches(patches, labels, k)
    sel_idx = {p.index for p in selected}
    for _, r in sub.iterrows():
        idx = int(r.get("patch_index", r.get("index", -1)))
        if idx in sel_idx and bool(r["label_positive"]):
            return True
    return False


def diagnose_record(rec: dict, label_df: pd.DataFrame, *, headline_k: int) -> dict:
    iid = _image_id(rec)
    answers = _answers(rec)
    boxes = load_cached_ocr_boxes(iid) or []
    full_text = _fullpage_text(boxes)
    answer_boxes = find_answer_boxes(boxes, answers)

    sub = label_df[label_df["image_id"] == iid]
    row = {
        "image_id": iid,
        "question_id": rec.get("question_id", ""),
        "question": rec.get("question", ""),
        "answers": " | ".join(answers),
        "fullpage_answer_present": answer_in_text(answers, full_text),
        "answer_boxes_on_fullpage": bool(answer_boxes),
        "candidate_evidence_reachability": bool(sub["label_positive"].any()) if not sub.empty else False,
        "candidate_ocr_exact_reachability": bool(sub["label_exact_patch_ocr"].any()) if not sub.empty else False,
        "fullpage_box_overlap_positive": bool(sub["label_fullpage_box_overlap"].any()) if not sub.empty else False,
        "soft_token_positive": bool(sub["label_soft_token_overlap"].any()) if not sub.empty else False,
        "fuzzy_positive": bool(sub["label_fuzzy_match"].any()) if not sub.empty else False,
    }
    if not sub.empty:
        row[f"oracle_evidence_at_{headline_k}"] = _oracle_label_hit(sub, headline_k)
        row[f"oracle_ocr_exact_at_{headline_k}"] = _oracle_patch_ocr_hit(
            sub, iid, answers, headline_k
        )
    else:
        row[f"oracle_evidence_at_{headline_k}"] = False
        row[f"oracle_ocr_exact_at_{headline_k}"] = False
    return row


def _rate(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if r.get(key)) / len(rows)


def _write_summary(rows: list[dict], out_path: Path, headline_k: int) -> list[dict]:
    metrics = [
        "fullpage_answer_present",
        "candidate_evidence_reachability",
        "candidate_ocr_exact_reachability",
        "fullpage_box_overlap_positive",
        "soft_token_positive",
        "fuzzy_positive",
        f"oracle_evidence_at_{headline_k}",
        f"oracle_ocr_exact_at_{headline_k}",
    ]
    summary = [{"metric": m, "rate": _rate(rows, m), "count": sum(1 for r in rows if r.get(m)), "n": len(rows)} for m in metrics]

    fp = _rate(rows, "fullpage_answer_present")
    oracle = _rate(rows, f"oracle_ocr_exact_at_{headline_k}")
    summary.append({
        "metric": "gap_fullpage_minus_oracle_ocr_exact",
        "rate": fp - oracle,
        "count": int(round((fp - oracle) * len(rows))),
        "n": len(rows),
    })

    buckets = [
        ("fullpage_yes_candidate_evidence_no", lambda r: r["fullpage_answer_present"] and not r["candidate_evidence_reachability"]),
        ("fullpage_yes_box_overlap_no", lambda r: r["fullpage_answer_present"] and not r["fullpage_box_overlap_positive"]),
        ("box_overlap_yes_ocr_exact_no", lambda r: r["fullpage_box_overlap_positive"] and not r["candidate_ocr_exact_reachability"]),
        ("candidate_evidence_yes_ocr_exact_no", lambda r: r["candidate_evidence_reachability"] and not r["candidate_ocr_exact_reachability"]),
        ("fullpage_yes_oracle_ocr_exact_no", lambda r: r["fullpage_answer_present"] and not r[f"oracle_ocr_exact_at_{headline_k}"]),
    ]
    for name, pred in buckets:
        count = sum(1 for r in rows if pred(r))
        summary.append({"metric": f"bucket_{name}", "rate": count / len(rows) if rows else 0.0, "count": count, "n": len(rows)})

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "rate", "count", "n"])
        w.writeheader()
        w.writerows(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose full-page vs oracle coverage gap.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--headline-k", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    logger = setup_experiment_logging("coverage_gap")
    log_section(logger, f"Coverage gap diagnostic | manifest={args.manifest}")

    labels_path = outputs_path("labels", "patch_labels.parquet")
    if not labels_path.exists():
        logger.error("Missing %s — run label_patches_from_answers.py first", labels_path)
        sys.exit(1)
    label_df = pd.read_parquet(labels_path)

    records = list(iter_manifest(args.manifest))
    if args.limit:
        records = records[: args.limit]

    rows = [diagnose_record(rec, label_df, headline_k=args.headline_k) for rec in records]

    per_q = outputs_path("metrics", "coverage_gap_per_question.csv")
    per_q.parent.mkdir(parents=True, exist_ok=True)
    with open(per_q, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info("Wrote %s (%d rows)", per_q, len(rows))

    summary_path = outputs_path("metrics", "coverage_gap_summary.csv")
    summary = _write_summary(rows, summary_path, args.headline_k)
    logger.info("Wrote %s", summary_path)

    for item in summary:
        if item["metric"] in {
            "fullpage_answer_present",
            "candidate_evidence_reachability",
            "candidate_ocr_exact_reachability",
            "fullpage_box_overlap_positive",
            "soft_token_positive",
            "fuzzy_positive",
            f"oracle_ocr_exact_at_{args.headline_k}",
            f"oracle_evidence_at_{args.headline_k}",
            "gap_fullpage_minus_oracle_ocr_exact",
        }:
            logger.info("  %s = %.3f (%d/%d)", item["metric"], item["rate"], item["count"], item["n"])


if __name__ == "__main__":
    main()
