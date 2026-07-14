#!/usr/bin/env python3
"""Weight/MMR sweep for QE-BOPS v2 top-2 precision tuning."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.data.dataset_loader import iter_manifest
from src.features.ocr_layout_graph import build_ocr_layout_graph
from src.features.patch_features import extract_patch_features
from src.metrics.coverage_eval import eval_selection
from src.metrics.answer_coverage import rank_candidates_by_score
from src.preprocessing.mmr_select import mmr_select
from src.preprocessing.patch_grid import Patch
from src.preprocessing.patch_scoring_qa import score_patch_question_aware
from src.preprocessing.qe_bops_scoring import QEBOPS_V2_WEIGHTS, _label_value_boost
from src.preprocessing.selectors import _build_fair_pool
from src.utils.image_io import load_image
from src.utils.logging_utils import log_section, setup_experiment_logging
from src.utils.ocr_cache import load_cached_ocr_boxes
from src.utils.paths import outputs_path

_SCORE_KEYS = (
    "bm25",
    "label_value_proximity",
    "same_row_label_value",
    "below_label_relation",
    "answer_type",
    "text_confidence",
    "edge_density",
    "entropy",
)


def _image_id(record: dict) -> str:
    return record.get("image_id") or record.get("doc_id") or record.get("question_id", "unknown")


def _answers(record: dict) -> list[str]:
    ans = record.get("answers") or record.get("answer") or []
    if isinstance(ans, str):
        return [ans]
    return list(ans)


def _load_patch_text_map(iid: str) -> dict[int, str]:
    path = outputs_path("ocr", "patches", f"{iid}.json")
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {int(p["index"]): p.get("text", "") for p in data.get("patches", [])}


def _make_weights(qa: float, lv: float, at: float) -> dict[str, float]:
    remain = 1.0 - qa
    w = dict(QEBOPS_V2_WEIGHTS)
    w["q_bops_score"] = qa
    w["label_value_proximity"] = lv * 0.5
    w["same_row_label_value"] = lv * 0.25
    w["below_label_relation"] = lv * 0.25
    w["answer_type"] = at
    tail = ["bm25", "text_confidence", "edge_density", "entropy"]
    tail_sum = sum(w[k] for k in tail)
    scale = (remain - lv - at) / tail_sum if tail_sum > 0 else 1.0
    for k in tail:
        w[k] *= scale
    return w


def _score_patch(feats: dict[str, float], boost: float, weights: dict[str, float]) -> float:
    score = weights["q_bops_score"] * feats["q_bops_score"]
    for key in _SCORE_KEYS:
        score += weights.get(key, 0.0) * feats.get(key, 0.0)
    return score + boost


@dataclass
class _RecordCache:
    pool: list[Patch]
    patch_items: list[tuple[Patch, dict[str, float], float]]
    labels: list[dict]
    answers: list[str]
    patch_texts: dict[int, str]


def _precompute_records(records: list[dict], labels_df: pd.DataFrame, logger) -> list[_RecordCache]:
    labels_by_image = {
        iid: [row.to_dict() for _, row in grp.iterrows()]
        for iid, grp in labels_df.groupby("image_id")
    }
    cached: list[_RecordCache] = []
    t0 = time.perf_counter()
    for idx, rec in enumerate(records, start=1):
        iid = _image_id(rec)
        question = rec.get("question", "")
        image = load_image(rec["image_path"])
        boxes = load_cached_ocr_boxes(iid) or []
        pool = _build_fair_pool(image, question, boxes)
        graph = build_ocr_layout_graph(boxes)
        patch_items: list[tuple[Patch, dict[str, float], float]] = []
        for patch in pool:
            feats = extract_patch_features(
                image, patch, boxes, question, layout_graph=graph,
            )
            feats["q_bops_score"] = score_patch_question_aware(image, patch, boxes, question)
            boost = _label_value_boost(question, patch, boxes)
            patch_items.append((patch, feats, boost))
        cached.append(_RecordCache(
            pool=pool,
            patch_items=patch_items,
            labels=labels_by_image.get(iid, []),
            answers=_answers(rec),
            patch_texts=_load_patch_text_map(iid),
        ))
        if idx == 1 or idx % 10 == 0 or idx == len(records):
            elapsed = time.perf_counter() - t0
            logger.info("Precomputed %d/%d images (%.1fs)", idx, len(records), elapsed)
    return cached


def _eval_config(
    cached: list[_RecordCache],
    weights: dict[str, float],
    *,
    k: int,
    mmr_lam: float | None,
) -> tuple[float, float]:
    strict_hits = any_hits = 0
    for rec in cached:
        scores = [_score_patch(feats, boost, weights) for _, feats, boost in rec.patch_items]
        pool = rec.pool
        if mmr_lam is None:
            ranked = sorted(zip(pool, scores), key=lambda x: x[1], reverse=True)
            selected = [p for p, _ in ranked[:k]]
        else:
            selected = mmr_select(pool, scores, k, lambda_=mmr_lam)
        sel_idx = {p.index for p in selected}
        texts = [rec.patch_texts.get(i, "") for i in sel_idx]
        ranked_idx = rank_candidates_by_score(pool, scores)
        ev = eval_selection(rec.labels, sel_idx, rec.answers, texts, ranked_idx)
        strict_hits += int(ev["evidence_strict"])
        any_hits += int(ev["evidence_any"])
    n = len(cached)
    return (strict_hits / n if n else 0.0, any_hits / n if n else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="QE-BOPS v2 weight sweep.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    logger = setup_experiment_logging("weight_sweep")
    log_section(logger, "QE-BOPS v2 weight sweep")

    labels_df = pd.read_parquet(outputs_path("labels", "patch_labels.parquet"))
    records = list(iter_manifest(args.manifest))
    if args.limit:
        records = records[: args.limit]

    q_anchors = [0.40, 0.50, 0.60, 0.70]
    lv_weights = [0.10, 0.15, 0.20, 0.25]
    at_weights = [0.05, 0.10, 0.15]
    mmr_opts: list[tuple[str, float | None]] = [
        ("no_mmr", None),
        ("mmr_0.75", 0.75),
        ("mmr_0.90", 0.90),
        ("mmr_1.00", 1.0),
    ]
    configs = list(itertools.product(q_anchors, lv_weights, at_weights, mmr_opts))
    logger.info(
        "Grid: %d configs (%d q-anchor x %d label-value x %d answer-type x %d MMR) on %d images",
        len(configs), len(q_anchors), len(lv_weights), len(at_weights), len(mmr_opts), len(records),
    )

    cached = _precompute_records(records, labels_df, logger)
    logger.info("Feature cache ready; sweeping weights...")

    rows = []
    out = outputs_path("metrics", "qe_bops_weight_sweep.csv")
    fieldnames = [
        "q_bops_anchor", "label_value_weight", "answer_type_weight", "mmr",
        "evidence_strict", "evidence_any", "n",
    ]
    t_sweep = time.perf_counter()
    for cfg_idx, (qa, lv, at, (mmr_name, mmr_lam)) in enumerate(configs, start=1):
        weights = _make_weights(qa, lv, at)
        strict, any_cov = _eval_config(cached, weights, k=args.k, mmr_lam=mmr_lam)
        row = {
            "q_bops_anchor": qa,
            "label_value_weight": lv,
            "answer_type_weight": at,
            "mmr": mmr_name,
            "evidence_strict": strict,
            "evidence_any": any_cov,
            "n": len(records),
        }
        rows.append(row)
        if cfg_idx == 1 or cfg_idx % 10 == 0 or cfg_idx == len(configs):
            logger.info(
                "Config %d/%d | strict=%.3f any=%.3f | qa=%.2f lv=%.2f at=%.2f %s (%.1fs)",
                cfg_idx, len(configs), strict, any_cov, qa, lv, at, mmr_name,
                time.perf_counter() - t_sweep,
            )

    rows.sort(key=lambda r: (r["evidence_strict"], r["evidence_any"]), reverse=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s (%d configs)", out, len(rows))
    best = rows[0]
    logger.info(
        "Best strict=%.3f any=%.3f | qa=%.2f lv=%.2f at=%.2f mmr=%s",
        best["evidence_strict"], best["evidence_any"],
        best["q_bops_anchor"], best["label_value_weight"],
        best["answer_type_weight"], best["mmr"],
    )


if __name__ == "__main__":
    main()
