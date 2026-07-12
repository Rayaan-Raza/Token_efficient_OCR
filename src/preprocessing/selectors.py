"""Fair patch selector methods for QE-BOPS evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from PIL import Image

from src.metrics.answer_coverage import oracle_select_patches
from src.preprocessing.bops import select_ocr_guided_patches, select_random_patches, select_uniform_patches
from src.preprocessing.candidate_pool import (
    CandidatePoolConfig,
    generate_candidate_pool,
    generate_original_grid,
    pool_hash,
)
from src.preprocessing.mmr_select import mmr_select
from src.preprocessing.patch_grid import Patch, crop_patch
from src.preprocessing.patch_scoring import score_patch, text_confidence_score
from src.preprocessing.patch_scoring_qa import score_patch_question_aware
from src.preprocessing.qe_bops_scoring import score_qe_bops_patch
from src.features.patch_features import bm25_score, answer_type_score, extract_patch_features


class AnswerLeakageError(ValueError):
    """Raised when GT answers are passed to inference-time selection."""


FAIR_METHODS = {
    "random",
    "uniform",
    "bops_fair_pool",
    "bops_qa_fair_pool",
    "ocr_confidence_topk",
    "bm25_only",
    "question_overlap_topk",
    "answer_type_only",
    "multiscale_uniform",
    "edge_strip",
    "qe_bops",
    "qe_bops_no_question",
}

DIAGNOSTIC_METHODS = {"oracle"}


@dataclass
class SelectionResult:
    method: str
    patches: list[Patch]
    patch_images: list[Image.Image]
    scores: list[float]
    candidate_pool: list[Patch]
    pool_hash_value: str
    meta: dict[str, Any]


def _reject_answers(method: str, answers: list[str] | None, eval_labels: bool) -> None:
    if answers and method not in DIAGNOSTIC_METHODS and not eval_labels:
        raise AnswerLeakageError(
            f"Ground-truth answers cannot be used for method={method} at inference time."
        )


def _build_fair_pool(
    image: Image.Image,
    question: str,
    ocr_boxes: list[dict[str, Any]],
    *,
    include_question_centered: bool = True,
) -> list[Patch]:
    cfg = CandidatePoolConfig(include_question_token_centered=include_question_centered)
    return generate_candidate_pool(
        image, question, ocr_boxes, cfg, include_question_centered=include_question_centered
    )


def select_patches(
    image: Image.Image,
    method: str,
    k: int,
    question: str,
    ocr_boxes: list[dict[str, Any]],
    *,
    seed: int = 0,
    answers: list[str] | None = None,
    patch_labels: list[dict[str, Any]] | None = None,
    eval_labels: bool = False,
    mmr_lambda: float = 0.75,
) -> SelectionResult:
    """Select K patches under fair-comparison protocol."""
    _reject_answers(method, answers, eval_labels)

    if method == "bops_original":
        candidates = generate_original_grid(image)
    elif method == "qe_bops_no_question":
        candidates = _build_fair_pool(image, question, ocr_boxes, include_question_centered=False)
    else:
        candidates = _build_fair_pool(image, question, ocr_boxes, include_question_centered=True)

    qid = question[:32]
    ph = pool_hash(candidates, qid)

    scores: list[float] = []
    selected: list[Patch] = []

    if method == "random":
        selected = select_random_patches(candidates, k, seed=seed)
    elif method == "uniform":
        selected = select_uniform_patches(candidates, k)
    elif method in ("bops_fair_pool", "bops_original"):
        selected, scores = select_ocr_guided_patches(image, candidates, k, ocr_boxes)
    elif method == "bops_qa_fair_pool":
        for p in candidates:
            scores.append(score_patch_question_aware(image, p, ocr_boxes, question))
        selected = mmr_select(candidates, scores, k, lambda_=0.5)
        scores = [scores[candidates.index(p)] for p in selected]
    elif method == "ocr_confidence_topk":
        scores = [text_confidence_score(p, ocr_boxes) for p in candidates]
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        selected = [p for p, _ in ranked[:k]]
    elif method == "bm25_only":
        from src.preprocessing.patch_scoring_qa import patch_ocr_text

        scores = [bm25_score(question, patch_ocr_text(p, ocr_boxes)) for p in candidates]
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        selected = [p for p, _ in ranked[:k]]
    elif method == "question_overlap_topk":
        from src.preprocessing.patch_scoring_qa import question_token_overlap_score

        scores = [question_token_overlap_score(p, ocr_boxes, question) for p in candidates]
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        selected = [p for p, _ in ranked[:k]]
    elif method == "answer_type_only":
        from src.preprocessing.patch_scoring_qa import patch_ocr_text

        scores = [answer_type_score(question, patch_ocr_text(p, ocr_boxes)) for p in candidates]
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        selected = [p for p, _ in ranked[:k]]
    elif method == "multiscale_uniform":
        selected = select_uniform_patches(candidates, k)
    elif method == "edge_strip":
        edge = [p for p in candidates if p.y <= 5 or p.x <= 5][:k]
        selected = edge if len(edge) >= k else select_uniform_patches(candidates, k)
    elif method == "qe_bops":
        scores = []
        for p in candidates:
            s, _ = score_qe_bops_patch(image, p, ocr_boxes, question, include_question=True)
            scores.append(s)
        selected = mmr_select(candidates, scores, k, lambda_=mmr_lambda)
        scores = [scores[candidates.index(p)] for p in selected]
    elif method == "qe_bops_no_question":
        scores = []
        for p in candidates:
            s, _ = score_qe_bops_patch(image, p, ocr_boxes, question, include_question=False)
            scores.append(s)
        selected = mmr_select(candidates, scores, k, lambda_=mmr_lambda)
        scores = [scores[candidates.index(p)] for p in selected]
    elif method == "oracle":
        if not patch_labels:
            raise ValueError("oracle selection requires patch_labels from eval pipeline")
        selected = oracle_select_patches(candidates, patch_labels, k)
        scores = [lbl["label_confidence"] for lbl in patch_labels[: len(selected)]]
    else:
        raise ValueError(f"Unknown method: {method}")

    patch_images = [crop_patch(image, p) for p in selected]
    meta = {
        "method": method,
        "num_candidates": len(candidates),
        "num_selected": len(selected),
        "pool_hash": ph,
    }
    return SelectionResult(method, selected, patch_images, scores, candidates, ph, meta)


def assert_same_pool_hash(a: SelectionResult, b: SelectionResult) -> None:
    if a.pool_hash_value != b.pool_hash_value:
        raise AssertionError(f"Pool hash mismatch: {a.method} vs {b.method}")
