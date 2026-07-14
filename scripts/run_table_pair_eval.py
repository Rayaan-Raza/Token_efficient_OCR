#!/usr/bin/env python3
"""Coverage eval for qe_bops_table_pair with margin sweep and error counts."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.data.dataset_loader import iter_manifest
from src.metrics.coverage_eval import eval_selection
from src.metrics.answer_coverage import rank_candidates_by_score
from src.metrics.gates import load_headline_k
from src.preprocessing.qe_bops_table_pair import select_table_pair_patches
from src.preprocessing.selectors import _build_fair_pool, select_patches
from src.utils.image_io import load_image
from src.utils.logging_utils import log_section, setup_experiment_logging
from src.utils.ocr_cache import load_cached_ocr_boxes
from src.utils.paths import outputs_path


def _image_id(record: dict) -> str:
    return record.get("image_id") or record.get("doc_id", "")


def _answers(record: dict) -> list[str]:
    ans = record.get("answers") or record.get("answer") or []
    if isinstance(ans, str):
        return [ans]
    return list(ans)


def _patch_texts(iid: str, indices: set[int]) -> list[str]:
    import json
    path = outputs_path("ocr", "patches", f"{iid}.json")
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [p.get("text", "") for p in data["patches"] if p.get("index") in indices]


def _eval_method(records, labels_df, method: str, hk: int, *, margin: float = 0.05) -> dict:
    rows = []
    for rec in records:
        iid = _image_id(rec)
        q = rec.get("question", "")
        answers = _answers(rec)
        image = load_image(rec["image_path"])
        boxes = load_cached_ocr_boxes(iid) or []
        labels = [r.to_dict() for _, r in labels_df[labels_df["image_id"] == iid].iterrows()]

        if method == "qe_bops_table_pair":
            pool = _build_fair_pool(image, q, boxes)
            patches, scores, all_scores, _ = select_table_pair_patches(
                image, pool, hk, q, boxes, swap_margin=margin,
            )
            sel_idx = {p.index for p in patches}
            ranked = rank_candidates_by_score(pool, all_scores)
        else:
            sel = select_patches(image, method, hk, q, boxes)
            sel_idx = {p.index for p in sel.patches}
            ranked = rank_candidates_by_score(sel.candidate_pool, sel.meta.get("candidate_scores", []))
            all_scores = sel.meta.get("candidate_scores", [])

        texts = _patch_texts(iid, sel_idx)
        ev = eval_selection(labels, sel_idx, answers, texts, ranked)
        rows.append({
            "image_id": iid,
            "evidence_strict": float(ev["evidence_strict"]),
            "evidence_any": float(ev["evidence_any"]),
            "ocr_exact_coverage": float(ev["ocr_exact_coverage"]),
            **{f"coverage_at_{kk}_{tier}": float(ev[f"coverage_at_{kk}_{tier}"])
               for kk in (1, 2, 4, 8) for tier in ("strict", "any")},
            "mean_rank_first_positive_any": ev["mean_rank_first_positive_any"],
        })

    df = pd.DataFrame(rows)
    out = {
        "method": method,
        "margin": margin,
        "n": len(df),
        "evidence_strict": df["evidence_strict"].mean(),
        "evidence_any": df["evidence_any"].mean(),
        "ocr_exact_coverage": df["ocr_exact_coverage"].mean(),
        "coverage_at_1_any": df["coverage_at_1_any"].mean(),
        "coverage_at_2_any": df["coverage_at_2_any"].mean(),
        "coverage_at_4_any": df["coverage_at_4_any"].mean(),
        "coverage_at_8_any": df["coverage_at_8_any"].mean(),
        "mean_rank_first_positive_any": df["mean_rank_first_positive_any"].dropna().mean(),
    }
    return out, df


def _error_counts(tp_df: pd.DataFrame, qa_df: pd.DataFrame) -> dict:
    merged = tp_df.merge(qa_df, on="image_id", suffixes=("_tp", "_qa"))
    return {
        "q_hit_tp_miss": int(((merged["evidence_any_qa"] > 0) & (merged["evidence_any_tp"] == 0)).sum()),
        "tp_hit_q_miss": int(((merged["evidence_any_tp"] > 0) & (merged["evidence_any_qa"] == 0)).sum()),
        "both_hit": int(((merged["evidence_any_tp"] > 0) & (merged["evidence_any_qa"] > 0)).sum()),
        "neither_hit": int(((merged["evidence_any_tp"] == 0) & (merged["evidence_any_qa"] == 0)).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Table-pair coverage eval + margin sweep.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--margins", default="0.0,0.05,0.08,0.10")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    logger = setup_experiment_logging("table_pair_eval")
    log_section(logger, "qe_bops_table_pair eval")

    records = list(iter_manifest(args.manifest))
    if args.limit:
        records = records[: args.limit]
    labels_df = pd.read_parquet(outputs_path("labels", "patch_labels.parquet"))
    hk = load_headline_k(2)

    margins = [float(x) for x in args.margins.split(",")]
    sweep_rows = []
    best_df: pd.DataFrame | None = None
    qa_df: pd.DataFrame | None = None

    for margin in margins:
        row, df = _eval_method(records, labels_df, "qe_bops_table_pair", hk, margin=margin)
        sweep_rows.append(row)
        logger.info(
            "margin=%.2f | strict=%.3f any=%.3f ocr_exact=%.3f rank=%.1f",
            margin, row["evidence_strict"], row["evidence_any"],
            row["ocr_exact_coverage"], row["mean_rank_first_positive_any"] or 0,
        )
        if margin == 0.05:
            best_df = df

    qa_summary, qa_df = _eval_method(records, labels_df, "bops_qa_fair_pool", hk)
    logger.info(
        "Q-BOPS | strict=%.3f any=%.3f ocr_exact=%.3f",
        qa_summary["evidence_strict"], qa_summary["evidence_any"], qa_summary["ocr_exact_coverage"],
    )

    v2_summary, _ = _eval_method(records, labels_df, "qe_bops_v2", hk)
    logger.info(
        "qe_bops_v2 | strict=%.3f any=%.3f",
        v2_summary["evidence_strict"], v2_summary["evidence_any"],
    )

    if best_df is not None and qa_df is not None:
        errs = _error_counts(best_df, qa_df)
        logger.info("Error counts (margin=0.05 vs Q-BOPS): %s", errs)

    out = outputs_path("metrics", "table_pair_margin_sweep.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        w.writeheader()
        w.writerows(sweep_rows)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
