"""Patch feature extraction for QE-BOPS heuristic and learned rankers."""

from __future__ import annotations

import math
import re
from typing import Any

from PIL import Image

from src.features.ocr_layout_graph import build_ocr_layout_graph, graph_features_for_patch
from src.ocr.normalize_text import normalize_text, tokenize
from src.preprocessing.patch_grid import Patch
from src.preprocessing.patch_scoring import (
    edge_density_score,
    entropy_score,
    text_confidence_score,
    text_coverage_score,
)
from src.preprocessing.patch_scoring_qa import patch_ocr_text, question_token_overlap_score, question_tokens


_DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b")
_CURRENCY_RE = re.compile(r"[$€£]\s?\d+|\d+\.\d{2}")
_PERCENT_RE = re.compile(r"\d+\s?%")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")


def bm25_score(question: str, doc: str, k1: float = 1.5, b: float = 0.75) -> float:
    qtoks = question_tokens(question)
    dtoks = tokenize(doc)
    if not qtoks or not dtoks:
        return 0.0
    dl = len(dtoks)
    avgdl = max(dl, 1.0)
    score = 0.0
    tf_map = {}
    for t in dtoks:
        tf_map[t] = tf_map.get(t, 0) + 1
    for t in qtoks:
        if t not in tf_map:
            continue
        tf = tf_map[t]
        num = tf * (k1 + 1)
        den = tf + k1 * (1 - b + b * dl / avgdl)
        score += num / den
    return score / len(qtoks)


def answer_type_score(question: str, patch_text: str) -> float:
    q = question.lower()
    t = patch_text.lower()
    if any(w in q for w in ("date", "when", "year", "month", "day")):
        return 1.0 if _DATE_RE.search(t) else 0.0
    if any(w in q for w in ("amount", "price", "cost", "total", "fee")):
        return 1.0 if _CURRENCY_RE.search(t) else 0.0
    if "percent" in q or "%" in q:
        return 1.0 if _PERCENT_RE.search(t) else 0.0
    if "email" in q:
        return 1.0 if _EMAIL_RE.search(t) else 0.0
    if any(w in q for w in ("yes", "no")):
        return 1.0 if any(w in t for w in ("yes", "no")) else 0.0
    return 0.0


def label_value_proximity_score(
    patch: Patch,
    ocr_boxes: list[dict[str, Any]],
    question: str,
) -> float:
    qtoks = set(question_tokens(question))
    if not qtoks:
        return 0.0
    best = 0.0
    label_boxes = []
    for b in ocr_boxes:
        text = normalize_text(b.get("text", ""))
        if any(t in text for t in qtoks) or ":" in b.get("text", ""):
            label_boxes.append(b)
    if not label_boxes:
        return 0.0
    for lb in label_boxes:
        lxs = [p[0] for p in lb["box"]]
        lys = [p[1] for p in lb["box"]]
        lx1 = max(lxs)
        ly = sum(lys) / len(lys)
        for b in ocr_boxes:
            if b is lb:
                continue
            xs = [p[0] for p in b["box"]]
            ys = [p[1] for p in b["box"]]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            if patch.x <= cx < patch.x + patch.w and patch.y <= cy < patch.y + patch.h:
                if abs(cy - ly) < 25 and cx >= lx1 - 10:
                    best = max(best, 1.0)
                if cy > ly and abs(cx - lx1) < 80:
                    best = max(best, 0.7)
    return best


def extract_patch_features(
    image: Image.Image,
    patch: Patch,
    ocr_boxes: list[dict[str, Any]],
    question: str,
    *,
    include_question: bool = True,
) -> dict[str, float]:
    """Full feature vector for one candidate patch."""
    from src.preprocessing.patch_scoring_qa import patch_ocr_text as qa_patch_text

    ptext = qa_patch_text(patch, ocr_boxes)
    graph = build_ocr_layout_graph(ocr_boxes)
    gfeat = graph_features_for_patch(
        graph, patch.x, patch.y, patch.w, patch.h, set(question_tokens(question)) if include_question else set()
    )
    h, w = image.height, image.width
    feats: dict[str, float] = {
        "text_coverage": text_coverage_score(patch, ocr_boxes),
        "text_confidence": text_confidence_score(patch, ocr_boxes),
        "edge_density": edge_density_score(image, patch),
        "entropy": entropy_score(image, patch),
        "layout_x": patch.x / max(1, w),
        "layout_y": patch.y / max(1, h),
        "layout_w": patch.w / max(1, w),
        "layout_h": patch.h / max(1, h),
        "ocr_token_count": float(len(tokenize(ptext))),
        "label_value_proximity": label_value_proximity_score(patch, ocr_boxes, question) if include_question else 0.0,
        **gfeat,
    }
    if include_question:
        feats["bm25"] = bm25_score(question, ptext)
        feats["question_overlap"] = question_token_overlap_score(patch, ocr_boxes, question)
        feats["answer_type"] = answer_type_score(question, ptext)
    else:
        feats["bm25"] = 0.0
        feats["question_overlap"] = 0.0
        feats["answer_type"] = 0.0
    return feats
