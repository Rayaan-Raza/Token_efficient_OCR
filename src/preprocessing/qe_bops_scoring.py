"""Heuristic QE-BOPS scoring."""

from __future__ import annotations

from typing import Any

from PIL import Image

from src.features.patch_features import bm25_score, extract_patch_features
from src.preprocessing.patch_grid import Patch
from src.preprocessing.patch_scoring_qa import patch_ocr_text, question_token_overlap_score, score_patch_question_aware


DEFAULT_WEIGHTS = {
    "text_coverage": 0.10,
    "text_confidence": 0.10,
    "edge_density": 0.05,
    "entropy": 0.05,
    "bm25": 0.10,
    "question_overlap": 0.10,
    "answer_type": 0.15,
    "label_value_proximity": 0.20,
    "same_row_label_value": 0.10,
    "below_label_relation": 0.05,
}

# V2: anchor on Q-BOPS signal; layout/graph as secondary cues.
QEBOPS_V2_WEIGHTS = {
    "q_bops_score": 0.40,
    "bm25": 0.20,
    "label_value_proximity": 0.10,
    "same_row_label_value": 0.10,
    "below_label_relation": 0.10,
    "answer_type": 0.10,
    "text_confidence": 0.05,
    "edge_density": 0.025,
    "entropy": 0.025,
}


def _label_value_boost(question: str, patch: Patch, ocr_boxes: list[dict[str, Any]]) -> float:
    """Boost patches with strong question overlap near label/value structure."""
    overlap = question_token_overlap_score(patch, ocr_boxes, question)
    if overlap < 0.5:
        return 0.0
    text = patch_ocr_text(patch, ocr_boxes).lower()
    if any(tok in text for tok in (":", "total", "date", "name", "amount", "no.", "#")):
        return 0.15
    return 0.05


def score_qe_bops_patch(
    image: Image.Image,
    patch: Patch,
    ocr_boxes: list[dict[str, Any]],
    question: str,
    *,
    include_question: bool = True,
    weights: dict[str, float] | None = None,
    version: str = "v1",
) -> tuple[float, dict[str, float]]:
    """Return heuristic score and feature dict."""
    feats = extract_patch_features(
        image, patch, ocr_boxes, question, include_question=include_question
    )
    if version == "v2":
        w = weights or QEBOPS_V2_WEIGHTS
        q_bops = score_patch_question_aware(image, patch, ocr_boxes, question)
        feats["q_bops_score"] = q_bops
        score = w["q_bops_score"] * q_bops
        for key in ("bm25", "label_value_proximity", "same_row_label_value", "below_label_relation",
                    "answer_type", "text_confidence", "edge_density", "entropy"):
            score += w.get(key, 0.0) * feats.get(key, 0.0)
        score += _label_value_boost(question, patch, ocr_boxes)
        return score, feats

    w = weights or DEFAULT_WEIGHTS
    score = sum(weight * feats.get(key, 0.0) for key, weight in w.items())
    return score, feats


def retrieve_twostage_pool(
    image: Image.Image,
    candidates: list[Patch],
    ocr_boxes: list[dict[str, Any]],
    question: str,
    *,
    top_qa: int = 20,
    top_bm25: int = 20,
    top_answer_type: int = 10,
) -> list[Patch]:
    """High-recall union for stage-2 QE-BOPS reranking."""
    from src.features.patch_features import answer_type_score

    if not candidates:
        return []

    qa_scored = sorted(
        ((p, score_patch_question_aware(image, p, ocr_boxes, question)) for p in candidates),
        key=lambda x: x[1], reverse=True,
    )
    bm25_scored = sorted(
        ((p, bm25_score(question, patch_ocr_text(p, ocr_boxes))) for p in candidates),
        key=lambda x: x[1], reverse=True,
    )
    at_scored = sorted(
        ((p, answer_type_score(question, patch_ocr_text(p, ocr_boxes))) for p in candidates),
        key=lambda x: x[1], reverse=True,
    )

    seen: set[int] = set()
    pool: list[Patch] = []
    for ranked in (qa_scored[:top_qa], bm25_scored[:top_bm25], at_scored[:top_answer_type]):
        for p, _ in ranked:
            if p.index not in seen:
                seen.add(p.index)
                pool.append(p)
    return pool if pool else list(candidates)
