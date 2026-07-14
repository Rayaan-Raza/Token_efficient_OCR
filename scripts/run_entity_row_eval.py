#!/usr/bin/env python3
"""Coverage eval for qe_bops_entity_row with margin sweep."""

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
from src.preprocessing.qe_bops_entity_row import select_entity_row_patches
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


def _eval(records, labels_df, method: str, hk: int, *, margin: float = 0.05) -> tuple[dict, pd.DataFrame]:
    rows = []
    for rec in records:
        iid = _image_id(rec)
        q, answers = rec.get("question", ""), _answers(rec)
        image = load_image(rec["image_path"])
        boxes = load_cached_ocr_boxes(iid) or []
        labels = [r.to_dict() for _, r in labels_df[labels_df["image_id"] == iid].iterrows()]

        if method == "qe_bops_entity_row":
            pool = _build_fair_pool(image, q, boxes)
            patches, _, all_scores, _ = select_entity_row_patches(
                image, pool, hk, q, boxes, swap_margin=margin,
            )
            sel_idx = {p.index for p in patches}
            ranked = rank_candidates_by_score(pool, all_scores)
        else:
            sel = select_patches(image, method, hk, q, boxes)
            sel_idx = {p.index for p in sel.patches}
            ranked = rank_candidates_by_score(sel.candidate_pool, sel.meta.get("candidate_scores", []))

        ev = eval_selection(labels, sel_idx, answers, _patch_texts(iid, sel_idx), ranked)
        rows.append({
            "image_id": iid,
            "evidence_strict": float(ev["evidence_strict"]),
            "evidence_any": float(ev["evidence_any"]),
            "ocr_exact_coverage": float(ev["ocr_exact_coverage"]),
        })

    df = pd.DataFrame(rows)
    summary = {
        "method": method,
        "margin": margin,
        "n": len(df),
        "evidence_strict": df["evidence_strict"].mean(),
        "evidence_any": df["evidence_any"].mean(),
        "ocr_exact_coverage": df["ocr_exact_coverage"].mean(),
    }
    return summary, df


def main() -> None:
    parser = argparse.ArgumentParser(description="Entity-row coverage eval.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--margins", default="0.0,0.03,0.05,0.08")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    logger = setup_experiment_logging("entity_row_eval")
    log_section(logger, "qe_bops_entity_row eval")

    records = list(iter_manifest(args.manifest))
    if args.limit:
        records = records[: args.limit]
    labels_df = pd.read_parquet(outputs_path("labels", "patch_labels.parquet"))
    hk = load_headline_k(2)
    margins = [float(x) for x in args.margins.split(",")]

    sweep = []
    ref_df: pd.DataFrame | None = None
    for margin in margins:
        s, df = _eval(records, labels_df, "qe_bops_entity_row", hk, margin=margin)
        sweep.append(s)
        logger.info(
            "margin=%.2f | strict=%.3f any=%.3f ocr_exact=%.3f",
            margin, s["evidence_strict"], s["evidence_any"], s["ocr_exact_coverage"],
        )
        if margin == 0.05:
            ref_df = df

    qa_s, qa_df = _eval(records, labels_df, "bops_qa_fair_pool", hk)
    logger.info(
        "Q-BOPS | strict=%.3f any=%.3f ocr_exact=%.3f",
        qa_s["evidence_strict"], qa_s["evidence_any"], qa_s["ocr_exact_coverage"],
    )

    if ref_df is not None:
        merged = ref_df.merge(qa_df, on="image_id", suffixes=("_er", "_qa"))
        logger.info(
            "Errors margin=0.05: q_hit_er_miss=%d er_hit_q_miss=%d both=%d neither=%d",
            int(((merged["evidence_any_qa"] > 0) & (merged["evidence_any_er"] == 0)).sum()),
            int(((merged["evidence_any_er"] > 0) & (merged["evidence_any_qa"] == 0)).sum()),
            int(((merged["evidence_any_er"] > 0) & (merged["evidence_any_qa"] > 0)).sum()),
            int(((merged["evidence_any_er"] == 0) & (merged["evidence_any_qa"] == 0)).sum()),
        )

    out = outputs_path("metrics", "entity_row_margin_sweep.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
        w.writeheader()
        w.writerows(sweep)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
