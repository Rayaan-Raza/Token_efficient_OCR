#!/usr/bin/env python3
"""Deep-dive on Q-BOPS-hit / QE-BOPS-miss cases for G3 failure analysis."""

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
from src.metrics.answer_coverage import rank_candidates_by_score, _label_index
from src.metrics.coverage_eval import eval_selection, first_positive_rank, patch_evidence_tiers
from src.preprocessing.patch_scoring_qa import score_patch_question_aware
from src.preprocessing.qe_bops_scoring import score_qe_bops_patch
from src.preprocessing.selectors import _build_fair_pool, select_patches
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


def _patch_texts(iid: str, indices: set[int]) -> list[str]:
    path = outputs_path("ocr", "patches", f"{iid}.json")
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [p.get("text", "") for p in data["patches"] if p.get("index") in indices]


def _first_positive_rank(labels: list[dict], ranked_indices: list[int], tier: str) -> int | None:
    labels_by_index = {_label_index(lbl): lbl for lbl in labels}
    rank = first_positive_rank(ranked_indices, labels_by_index, tier)
    return int(rank) if rank is not None else None


def _classify_pattern(
    q_texts: list[str],
    qe_texts: list[str],
    qa_rank_any: int | None,
    qe_rank_any: int | None,
    qe_feats: dict,
) -> str:
    if qa_rank_any and qa_rank_any > 2 and (qe_rank_any is None or qe_rank_any > 2):
        return "both_deep_rank"
    if qa_rank_any and qa_rank_any <= 2 and qe_rank_any and qe_rank_any > 2:
        return "qe_correct_patch_not_top2"
    if qe_feats.get("label_value_proximity", 0) >= 0.7 and qe_feats.get("question_overlap", 0) < 0.5:
        return "label_only_no_question_overlap"
    if qe_feats.get("question_overlap", 0) < 0.3 and qe_feats.get("bm25", 0) < 0.3:
        return "layout_over_lexical"
    if q_texts and qe_texts and q_texts[0][:40] == qe_texts[0][:40]:
        return "wrong_second_patch_same_first"
    return "lexical_neighbor_miss"


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine Q-BOPS-hit / QE-miss failure cases.")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    logger = setup_experiment_logging("mine_q_miss")
    log_section(logger, "Q-BOPS-hit / QE-miss case mining")

    labels_df = pd.read_parquet(outputs_path("labels", "patch_labels.parquet"))
    records = { _image_id(r): r for r in iter_manifest(args.manifest) }

    miss_path = outputs_path("debug", "random_hits_qebops_misses", "q_bops_hit_qe_miss.csv")
    if not miss_path.exists():
        raise FileNotFoundError(f"Run analyze_g3_diagnostics.py first: {miss_path}")

    rows_out = []
    with open(miss_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            iid = row["image_id"]
            rec = records[iid]
            image = load_image(rec["image_path"])
            boxes = load_cached_ocr_boxes(iid) or []
            labels = [r.to_dict() for _, r in labels_df[labels_df["image_id"] == iid].iterrows()]
            q = rec.get("question", "")

            sel_qa = select_patches(image, "bops_qa_fair_pool", 2, q, boxes)
            sel_qe = select_patches(image, "qe_bops", 2, q, boxes)
            sel_anchor = select_patches(image, "qe_bops_anchor_pair", 2, q, boxes)

            pool = _build_fair_pool(image, q, boxes)
            qa_scores = [score_patch_question_aware(image, p, boxes, q) for p in pool]
            qe_scores = []
            qe_feats_best: dict = {}
            best = float("-inf")
            for p in pool:
                s, feats = score_qe_bops_patch(image, p, boxes, q, version="v2")
                qe_scores.append(s)
                if s >= best:
                    best = s
                    qe_feats_best = feats

            qa_ranked = rank_candidates_by_score(pool, qa_scores)
            qe_ranked = rank_candidates_by_score(pool, qe_scores)
            qa_rank_any = _first_positive_rank(labels, qa_ranked, "any")
            qe_rank_any = _first_positive_rank(labels, qe_ranked, "any")

            idx_anchor = {p.index for p in sel_anchor.patches}
            ev_anchor = eval_selection(
                labels, idx_anchor, _answers(rec), _patch_texts(iid, idx_anchor), qe_ranked,
            )

            idx_qa = {p.index for p in sel_qa.patches}
            idx_qe = {p.index for p in sel_qe.patches}
            q_texts = _patch_texts(iid, idx_qa)
            qe_texts = _patch_texts(iid, idx_qe)

            pattern = _classify_pattern(q_texts, qe_texts, qa_rank_any, qe_rank_any, qe_feats_best)
            rows_out.append({
                "image_id": iid,
                "question": q[:80],
                "pattern": pattern,
                "qa_rank_first_positive_any": qa_rank_any,
                "qe_rank_first_positive_any": qe_rank_any,
                "anchor_hit_any": ev_anchor["evidence_any"],
                "anchor_hit_strict": ev_anchor["evidence_strict"],
                "q_bops_top1_index": sel_qa.patches[0].index if sel_qa.patches else None,
                "qe_bops_top1_index": sel_qe.patches[0].index if sel_qe.patches else None,
                "anchor_top1_index": sel_anchor.patches[0].index if sel_anchor.patches else None,
                "anchor_top2_index": sel_anchor.patches[1].index if len(sel_anchor.patches) > 1 else None,
            })
            logger.info(
                "%s | pattern=%s | qa_rank=%s qe_rank=%s | anchor strict=%s any=%s",
                iid, pattern, qa_rank_any, qe_rank_any,
                ev_anchor["evidence_strict"], ev_anchor["evidence_any"],
            )

    out = outputs_path("debug", "random_hits_qebops_misses", "case_mining_detail.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    patterns: dict[str, int] = {}
    anchor_fixes = sum(1 for r in rows_out if r["anchor_hit_any"])
    for r in rows_out:
        patterns[r["pattern"]] = patterns.get(r["pattern"], 0) + 1

    summary = {
        "n_cases": len(rows_out),
        "anchor_pair_fixes_any": anchor_fixes,
        "pattern_counts": patterns,
    }
    summary_path = outputs_path("debug", "random_hits_qebops_misses", "case_mining_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote %s and %s", out, summary_path)
    logger.info("Anchor pair fixes %d/%d miss cases (any)", anchor_fixes, len(rows_out))


if __name__ == "__main__":
    main()
