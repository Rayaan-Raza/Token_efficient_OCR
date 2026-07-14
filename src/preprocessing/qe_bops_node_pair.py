"""QE-BOPS v3: OCR-token-first evidence retrieval with label-value pair selection."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any

from PIL import Image

from src.features.ocr_layout_graph import OCRLayoutGraph, OCRNode, build_ocr_layout_graph
from src.features.patch_features import answer_type_score, bm25_score
from src.ocr.normalize_text import normalize_text, tokenize
from src.preprocessing.mmr_select import patch_iou
from src.preprocessing.patch_grid import Patch
from src.preprocessing.patch_scoring_qa import patch_ocr_text, question_tokens, score_patch_question_aware


_REL_WEIGHTS = {
    "same_row": 1.0,
    "nearby_right": 0.85,
    "nearby_below": 0.80,
    "same_column": 0.65,
}

_LABEL_HINTS = (":", "total", "date", "name", "amount", "no.", "#", "held", "attended")
_NUMERIC_RE = re.compile(r"\d[\d,./%-]*")


@dataclass
class NodePair:
    label: OCRNode
    value: OCRNode
    relation: str
    score: float


def _trigrams(text: str) -> set[str]:
    t = normalize_text(text).replace(" ", "")
    if len(t) < 3:
        return {t} if t else set()
    return {t[i : i + 3] for i in range(len(t) - 2)}


def _token_similarity(a: str, b: str) -> float:
    na, nb = normalize_text(a), normalize_text(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        return 0.85
    tg_a, tg_b = _trigrams(na), _trigrams(nb)
    if not tg_a or not tg_b:
        return 0.0
    overlap = len(tg_a & tg_b) / max(1, len(tg_a | tg_b))
    return max(overlap, float(difflib.SequenceMatcher(None, na, nb).ratio()) * 0.75)


def late_interaction_score(question: str, text: str) -> float:
    """Max-over-token similarity per question token (cheap late interaction)."""
    qtoks = question_tokens(question)
    dtoks = tokenize(text)
    if not qtoks or not dtoks:
        return 0.0
    total = 0.0
    for qt in qtoks:
        best = max(_token_similarity(qt, dt) for dt in dtoks)
        total += best
    return total / len(qtoks)


def _node_in_patch(node: OCRNode, patch: Patch) -> bool:
    return patch.x <= node.cx < patch.x + patch.w and patch.y <= node.cy < patch.y + patch.h


def _is_label_like(node: OCRNode, question: str) -> bool:
    text = node.text.lower()
    qtoks = question_tokens(question)
    if any(t in text for t in qtoks):
        return True
    return any(h in text for h in _LABEL_HINTS)


def _is_value_like(node: OCRNode, question: str) -> bool:
    text = node.text.strip()
    if not text:
        return False
    if answer_type_score(question, text) > 0:
        return True
    if _NUMERIC_RE.search(text):
        return 1 <= len(tokenize(text)) <= 6
    return len(tokenize(text)) <= 4 and node.confidence >= 0.0


def _node_score(node: OCRNode, question: str) -> float:
    late = late_interaction_score(question, node.text)
    at = answer_type_score(question, node.text)
    bm = bm25_score(question, node.text)
    conf = min(1.0, max(0.0, node.confidence))
    label_bonus = 0.10 if _is_label_like(node, question) else 0.0
    value_bonus = 0.10 if _is_value_like(node, question) else 0.0
    return 0.40 * late + 0.25 * bm + 0.15 * at + 0.10 * conf + label_bonus + value_bonus


def _pair_score(label: OCRNode, value: OCRNode, relation: str, question: str) -> float:
    q_rel = max(
        late_interaction_score(question, label.text),
        late_interaction_score(question, f"{label.text} {value.text}"),
    )
    at = answer_type_score(question, value.text)
    rel_w = _REL_WEIGHTS.get(relation, 0.5)
    conf = min(1.0, max(0.0, value.confidence))
    lv_bonus = 0.12 if _is_label_like(label, question) and _is_value_like(value, question) else 0.0
    return 0.35 * q_rel + 0.25 * at + 0.20 * rel_w + 0.10 * conf + 0.10 * bm25_score(question, value.text) + lv_bonus


def enumerate_node_pairs(graph: OCRLayoutGraph, question: str) -> list[NodePair]:
    """Enumerate label-value OCR node pairs connected in the layout graph."""
    pairs: list[NodePair] = []
    seen: set[tuple[int, int]] = set()

    label_nodes = [n for n in graph.nodes if _is_label_like(n, question)]
    if not label_nodes:
        label_nodes = sorted(graph.nodes, key=lambda n: _node_score(n, question), reverse=True)[:12]

    for label in label_nodes:
        for nb_id, rel in graph.neighbors(label.node_id):
            if rel not in _REL_WEIGHTS:
                continue
            value = graph.nodes[nb_id]
            key = (min(label.node_id, value.node_id), max(label.node_id, value.node_id))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(NodePair(label, value, rel, _pair_score(label, value, rel, question)))

        for value in graph.nodes:
            if value.node_id == label.node_id:
                continue
            key = (min(label.node_id, value.node_id), max(label.node_id, value.node_id))
            if key in seen or not _is_value_like(value, question):
                continue
            dy = abs(label.cy - value.cy)
            dx = abs(label.cx - value.cx)
            rel = None
            if dy < 20 and 0 < value.cx - label.cx < 220:
                rel = "nearby_right"
            elif dx < 50 and 0 < value.cy - label.cy < 90:
                rel = "nearby_below"
            elif dy < 18:
                rel = "same_row"
            if rel is None:
                continue
            seen.add(key)
            pairs.append(NodePair(label, value, rel, _pair_score(label, value, rel, question)))

    pairs.sort(key=lambda p: p.score, reverse=True)
    return pairs


def _pair_support_in_patch(pair: NodePair, patch: Patch) -> float:
    label_in = _node_in_patch(pair.label, patch)
    value_in = _node_in_patch(pair.value, patch)
    if label_in and value_in:
        return 1.0
    if label_in or value_in:
        return 0.65
    return 0.0


def score_pool_patch_evidence(
    image: Image.Image,
    patch: Patch,
    graph: OCRLayoutGraph,
    pairs: list[NodePair],
    question: str,
    ocr_boxes: list[dict[str, Any]],
) -> float:
    """Node-first evidence score mapped onto a fair-pool patch."""
    nodes_in = [n for n in graph.nodes if _node_in_patch(n, patch)]
    pair_part = 0.0
    for pair in pairs[:24]:
        support = _pair_support_in_patch(pair, patch)
        if support > 0:
            pair_part = max(pair_part, pair.score * support)

    node_part = max((_node_score(n, question) for n in nodes_in), default=0.0)
    text = patch_ocr_text(patch, ocr_boxes)
    late = late_interaction_score(question, text)
    qa = score_patch_question_aware(image, patch, ocr_boxes, question)
    at = answer_type_score(question, text)
    lv_bonus = 0.10 if pair_part > 0.5 and late > 0.3 else 0.0
    return 0.30 * pair_part + 0.30 * qa + 0.20 * late + 0.10 * node_part + 0.10 * at + lv_bonus


def select_node_pair_patches(
    image: Image.Image,
    pool: list[Patch],
    k: int,
    question: str,
    ocr_boxes: list[dict[str, Any]],
) -> tuple[list[Patch], list[float], list[float], dict[str, Any]]:
    """Retrieve OCR evidence nodes/pairs, rerank fair-pool patches, select top-K."""
    graph = build_ocr_layout_graph(ocr_boxes)
    meta: dict[str, Any] = {}
    if not graph.nodes or not pool:
        return [], [], [], meta

    pairs = enumerate_node_pairs(graph, question)
    meta["node_pair_count"] = len(pairs)
    if pairs:
        meta["best_pair_score"] = pairs[0].score
        meta["best_pair_relation"] = pairs[0].relation

    qa_scores = {p.index: score_patch_question_aware(image, p, ocr_boxes, question) for p in pool}
    first = max(pool, key=lambda p: (qa_scores[p.index], -p.index))

    evidence_scored = sorted(
        ((p, score_pool_patch_evidence(image, p, graph, pairs, question, ocr_boxes)) for p in pool),
        key=lambda x: (x[1], -x[0].index),
        reverse=True,
    )

    selected: list[Patch] = [first]
    scores: list[float] = [qa_scores[first.index]]
    meta["slot1_source"] = "q_bops_anchor"

    for patch, ev_score in evidence_scored:
        if len(selected) >= k:
            break
        if patch.index == first.index:
            continue
        if patch_iou(patch, first) >= 0.35:
            continue
        selected.append(patch)
        scores.append(ev_score)
        meta["slot2_source"] = "node_pair_evidence"
        break

    for patch, ev_score in evidence_scored:
        if len(selected) >= k:
            break
        if patch.index in {s.index for s in selected}:
            continue
        selected.append(patch)
        scores.append(ev_score)

    all_scores = [ev for _, ev in evidence_scored]
    return selected[:k], scores[:k], all_scores, meta
