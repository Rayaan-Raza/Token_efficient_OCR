"""Question-aware patch scoring for BOPS (uses question only, never GT answers)."""

from __future__ import annotations

import difflib
from typing import Any

from src.ocr.normalize_text import normalize_text, tokenize
from src.preprocessing.patch_grid import Patch
from src.preprocessing.patch_scoring import (
    edge_density_score,
    entropy_score,
    text_confidence_score,
    text_coverage_score,
    _intersection_area,
)

_QUESTION_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "is", "it", "its",
    "of", "on", "or", "that", "the", "this", "to", "was", "were", "what",
    "when", "where", "which", "who", "why", "with",
})


def question_tokens(question: str) -> list[str]:
    """Content tokens from the question (no stopwords, length > 1)."""
    return [t for t in tokenize(question) if t not in _QUESTION_STOPWORDS and len(t) > 1]


def patch_ocr_text(patch: Patch, ocr_boxes: list[dict[str, Any]]) -> str:
    """Concatenate OCR text from boxes overlapping the patch."""
    texts: list[str] = []
    for b in ocr_boxes:
        if _intersection_area(patch, b["box"]) > 0:
            text = b.get("text", "").strip()
            if text:
                texts.append(text)
    return " ".join(texts)


def question_token_overlap_score(
    patch: Patch,
    ocr_boxes: list[dict[str, Any]],
    question: str,
) -> float:
    """Fraction of question content tokens found in patch OCR text."""
    qtoks = question_tokens(question)
    if not qtoks:
        return 0.0
    ptoks = set(tokenize(patch_ocr_text(patch, ocr_boxes)))
    if not ptoks:
        return 0.0
    hits = sum(1 for t in qtoks if t in ptoks)
    return hits / len(qtoks)


def semantic_similarity_score(
    patch: Patch,
    ocr_boxes: list[dict[str, Any]],
    question: str,
) -> float:
    """Normalized string similarity between question and patch OCR text."""
    q = normalize_text(question)
    p = normalize_text(patch_ocr_text(patch, ocr_boxes))
    if not q or not p:
        return 0.0
    return float(difflib.SequenceMatcher(None, q, p).ratio())


def score_patch_question_aware(
    image,
    patch: Patch,
    ocr_boxes: list[dict[str, Any]],
    question: str,
    weights: dict[str, float] | None = None,
) -> float:
    """Question-aware patch score (no ground-truth answer used)."""
    w = weights or {
        "text_coverage": 0.30,
        "text_confidence": 0.20,
        "question_overlap": 0.25,
        "semantic_similarity": 0.15,
        "edge_density": 0.05,
        "entropy": 0.05,
    }
    return (
        w["text_coverage"] * text_coverage_score(patch, ocr_boxes)
        + w["text_confidence"] * text_confidence_score(patch, ocr_boxes)
        + w["question_overlap"] * question_token_overlap_score(patch, ocr_boxes, question)
        + w["semantic_similarity"] * semantic_similarity_score(patch, ocr_boxes, question)
        + w["edge_density"] * edge_density_score(image, patch)
        + w["entropy"] * entropy_score(image, patch)
    )
