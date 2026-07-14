"""Feature schema and extraction for learned evidence rankers.

Labels / answers are never included in the feature matrix.
"""

from __future__ import annotations

import re
from typing import Any

from PIL import Image

from src.features.ocr_layout_graph import OCRLayoutGraph, build_ocr_layout_graph
from src.features.patch_features import extract_patch_features
from src.ocr.normalize_text import tokenize
from src.preprocessing.patch_grid import Patch
from src.preprocessing.patch_scoring_qa import patch_ocr_text, question_tokens, score_patch_question_aware
from src.preprocessing.qe_bops_node_pair import late_interaction_score

# Columns that must never appear as model inputs.
FORBIDDEN_FEATURE_COLUMNS = frozenset({
    "answer",
    "gt_answer",
    "answers",
    "label_exact_patch_ocr",
    "label_fullpage_box_overlap",
    "label_soft_token_overlap",
    "label_fuzzy_match",
    "label_positive",
    "strict_positive",
    "any_positive",
    "label_confidence",
})

# Stable ordered feature list for training / inference.
FEATURE_KEYS = [
    "text_coverage",
    "text_confidence",
    "edge_density",
    "entropy",
    "layout_x",
    "layout_y",
    "layout_w",
    "layout_h",
    "ocr_token_count",
    "bm25",
    "question_overlap",
    "answer_type",
    "label_value_proximity",
    "same_row_label_value",
    "below_label_relation",
    "dist_qnode_to_patch",
    "patch_line_density",
    "patch_box_density",
    "late_interaction",
    "q_bops_score",
    "q_bops_rank",
    "is_q_bops_top1",
    "is_q_bops_top2",
    "contains_q_node",
    "contains_value_like",
    "contains_label_like",
    "contains_label_and_value",
    "row_entity_score",
]

META_COLUMNS = frozenset({
    "image_id",
    "question_id",
    "group_id",
    "patch_index",
    "x",
    "y",
    "w",
    "h",
    "split",
    "question",
})

_LABEL_HINTS = (":", "total", "date", "name", "amount", "no.", "#", "held", "attended")
_NUMERIC_RE = re.compile(r"\d[\d,./%-]*")
_VALUE_LIKE_RE = re.compile(r"[$€£]?\d[\d,./%-]*|yes|no|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", re.I)


def assert_no_feature_leakage(feature_names: list[str] | set[str]) -> None:
    overlap = set(feature_names) & FORBIDDEN_FEATURE_COLUMNS
    if overlap:
        raise ValueError(f"Forbidden label/answer columns in feature matrix: {sorted(overlap)}")


def _node_in_patch(cx: float, cy: float, patch: Patch) -> bool:
    return patch.x <= cx < patch.x + patch.w and patch.y <= cy < patch.y + patch.h


def _is_label_like_text(text: str, qtoks: list[str]) -> bool:
    low = text.lower()
    if any(t in low for t in qtoks):
        return True
    return any(h in low for h in _LABEL_HINTS)


def _is_value_like_text(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if _VALUE_LIKE_RE.search(t):
        return True
    if _NUMERIC_RE.search(t) and 1 <= len(tokenize(t)) <= 6:
        return True
    return len(tokenize(t)) <= 4


def _layout_indicators(
    graph: OCRLayoutGraph,
    patch: Patch,
    question: str,
) -> dict[str, float]:
    qtoks = question_tokens(question)
    has_q = 0.0
    has_label = 0.0
    has_value = 0.0
    best_row = 0.0
    for n in graph.nodes:
        if not _node_in_patch(n.cx, n.cy, patch):
            continue
        low = n.text.lower()
        if qtoks and any(t in low for t in qtoks):
            has_q = 1.0
        if _is_label_like_text(n.text, qtoks):
            has_label = 1.0
        if _is_value_like_text(n.text):
            has_value = 1.0
        # Cheap row/entity cue: question token + value-like token both in patch
        if has_q and has_value:
            best_row = max(best_row, 1.0)
        elif has_label and has_value:
            best_row = max(best_row, 0.7)
    return {
        "contains_q_node": has_q,
        "contains_label_like": has_label,
        "contains_value_like": has_value,
        "contains_label_and_value": 1.0 if has_label and has_value else 0.0,
        "row_entity_score": best_row,
    }


def extract_ranker_features_for_patches(
    image: Image.Image,
    patches: list[Patch],
    ocr_boxes: list[dict[str, Any]],
    question: str,
    *,
    layout_graph: OCRLayoutGraph | None = None,
) -> list[dict[str, float]]:
    """Extract FEATURE_KEYS for each patch, including Q-BOPS score/rank flags."""
    graph = layout_graph if layout_graph is not None else build_ocr_layout_graph(ocr_boxes)
    scored: list[tuple[Patch, dict[str, float], float]] = []
    for patch in patches:
        feats = extract_patch_features(
            image, patch, ocr_boxes, question, include_question=True, layout_graph=graph
        )
        ptext = patch_ocr_text(patch, ocr_boxes)
        feats["late_interaction"] = late_interaction_score(question, ptext)
        q_score = float(score_patch_question_aware(image, patch, ocr_boxes, question))
        feats["q_bops_score"] = q_score
        feats.update(_layout_indicators(graph, patch, question))
        scored.append((patch, feats, q_score))

    # Rank by Q-BOPS score within this question's candidate set (1 = best).
    order = sorted(range(len(scored)), key=lambda i: scored[i][2], reverse=True)
    rank_of = {idx: rank + 1 for rank, idx in enumerate(order)}

    out: list[dict[str, float]] = []
    for i, (_patch, feats, _qs) in enumerate(scored):
        rank = rank_of[i]
        feats["q_bops_rank"] = float(rank)
        feats["is_q_bops_top1"] = 1.0 if rank == 1 else 0.0
        feats["is_q_bops_top2"] = 1.0 if rank <= 2 else 0.0
        # Ensure all keys present
        row = {k: float(feats.get(k, 0.0)) for k in FEATURE_KEYS}
        assert_no_feature_leakage(row.keys())
        out.append(row)
    return out


def feature_matrix_from_rows(rows: list[dict[str, float]]) -> list[list[float]]:
    assert_no_feature_leakage(FEATURE_KEYS)
    return [[float(r.get(k, 0.0)) for k in FEATURE_KEYS] for r in rows]
