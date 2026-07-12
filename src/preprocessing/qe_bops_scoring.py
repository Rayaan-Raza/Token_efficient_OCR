"""Heuristic QE-BOPS scoring."""

from __future__ import annotations

from typing import Any

from PIL import Image

from src.features.patch_features import extract_patch_features
from src.preprocessing.patch_grid import Patch


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


def score_qe_bops_patch(
    image: Image.Image,
    patch: Patch,
    ocr_boxes: list[dict[str, Any]],
    question: str,
    *,
    include_question: bool = True,
    weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Return heuristic score and feature dict."""
    w = weights or DEFAULT_WEIGHTS
    feats = extract_patch_features(
        image, patch, ocr_boxes, question, include_question=include_question
    )
    score = 0.0
    for key, weight in w.items():
        score += weight * feats.get(key, 0.0)
    return score, feats
