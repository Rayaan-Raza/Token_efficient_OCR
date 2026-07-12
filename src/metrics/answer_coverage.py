"""Answer coverage metrics and hierarchical weak labels for QE-BOPS."""

from __future__ import annotations

import difflib
import re
from typing import Any

from src.ocr.normalize_text import normalize_text, tokenize
from src.preprocessing.patch_grid import Patch
from src.preprocessing.patch_scoring import _intersection_area


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def answer_in_text(answers: list[str], text: str) -> bool:
    norm_text = _normalize_for_match(text)
    if not norm_text:
        return False
    for ans in answers:
        norm_ans = _normalize_for_match(ans)
        if norm_ans and norm_ans in norm_text:
            return True
    return False


def fuzzy_anls(a: str, b: str) -> float:
    na, nb = _normalize_for_match(a), _normalize_for_match(b)
    if not na or not nb:
        return 0.0
    return float(difflib.SequenceMatcher(None, na, nb).ratio())


def token_overlap_ratio(answers: list[str], text: str) -> float:
    ptoks = set(tokenize(text))
    if not ptoks:
        return 0.0
    best = 0.0
    for ans in answers:
        atoks = tokenize(ans)
        if not atoks:
            continue
        hits = sum(1 for t in atoks if t in ptoks)
        best = max(best, hits / len(atoks))
    return best


def _box_center_in_patch(box: list[list[int]], patch: Patch) -> bool:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    return patch.x <= cx < patch.x + patch.w and patch.y <= cy < patch.y + patch.h


def _box_overlap_ratio(box: list[list[int]], patch: Patch) -> float:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    bx0, bx1 = min(xs), max(xs)
    by0, by1 = min(ys), max(ys)
    ix0 = max(bx0, patch.x)
    iy0 = max(by0, patch.y)
    ix1 = min(bx1, patch.x + patch.w)
    iy1 = min(by1, patch.y + patch.h)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    box_area = max(1.0, (bx1 - bx0) * (by1 - by0))
    return inter / box_area


def find_answer_boxes(fullpage_boxes: list[dict[str, Any]], answers: list[str]) -> list[dict[str, Any]]:
    """Return OCR boxes whose text contains any normalized answer."""
    matched: list[dict[str, Any]] = []
    for b in fullpage_boxes:
        if answer_in_text(answers, b.get("text", "")):
            matched.append(b)
    return matched


def compute_patch_labels(
    patch_ocr_text: str,
    patch: Patch,
    answers: list[str],
    answer_boxes: list[dict[str, Any]],
    *,
    soft_token_min: float = 0.70,
    fuzzy_min: float = 0.50,
    box_overlap_min: float = 0.50,
) -> dict[str, Any]:
    """Hierarchical weak labels + label_confidence (eval/training only)."""
    exact = answer_in_text(answers, patch_ocr_text)
    box_overlap = False
    for b in answer_boxes:
        if _box_center_in_patch(b["box"], patch):
            box_overlap = True
            break
        if _box_overlap_ratio(b["box"], patch) >= box_overlap_min:
            box_overlap = True
            break

    soft = token_overlap_ratio(answers, patch_ocr_text) >= soft_token_min
    fuzzy = max(fuzzy_anls(a, patch_ocr_text) for a in answers) >= fuzzy_min if answers else False

    positive = exact or box_overlap or soft or fuzzy
    conf = 0.0
    if exact:
        conf = max(conf, 1.0)
    if box_overlap:
        conf = max(conf, 0.9)
    if soft:
        conf = max(conf, 0.7)
    if fuzzy:
        conf = max(conf, 0.5)

    return {
        "label_exact_patch_ocr": exact,
        "label_fullpage_box_overlap": box_overlap,
        "label_soft_token_overlap": soft,
        "label_fuzzy_match": fuzzy,
        "label_positive": positive,
        "label_confidence": conf,
    }


def coverage_at_k(selected_labels: list[bool]) -> float:
    """Fraction of samples where any of top-K patches is positive."""
    return float(any(selected_labels))


def oracle_select_patches(
    patches: list[Patch],
    labels: list[dict[str, Any]],
    k: int,
) -> list[Patch]:
    """Upper-bound oracle: pick top-K by label_confidence then label_positive."""
    scored = sorted(
        zip(patches, labels),
        key=lambda x: (x[1]["label_confidence"], x[1]["label_positive"]),
        reverse=True,
    )
    return [p for p, _ in scored[:k]]


def merged_patch_ocr_texts(texts: list[str]) -> str:
    return " ".join(t for t in texts if t)


def answer_in_selected_patches(answers: list[str], patch_texts: list[str]) -> bool:
    merged = merged_patch_ocr_texts(patch_texts)
    return answer_in_text(answers, merged)


def patch_from_dict(d: dict[str, Any]) -> Patch:
    return Patch(
        x=int(d["x"]),
        y=int(d["y"]),
        w=int(d["w"]),
        h=int(d["h"]),
        index=int(d.get("index", 0)),
    )
