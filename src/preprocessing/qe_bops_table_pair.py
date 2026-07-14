"""QE-BOPS table-row clustering + hard label-value co-occurrence selection."""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Any

from PIL import Image

from src.features.patch_features import answer_type_score, bm25_score
from src.ocr.normalize_text import normalize_text, tokenize
from src.preprocessing.mmr_select import mmr_select, patch_iou
from src.preprocessing.patch_grid import Patch
from src.preprocessing.patch_scoring_qa import (
    patch_ocr_text,
    question_token_overlap_score,
    question_tokens,
    score_patch_question_aware,
)
from src.preprocessing.qe_bops_node_pair import late_interaction_score


_NUMERIC_RE = re.compile(r"\d[\d,./%-]*")
_LABEL_HINTS = (":", "total", "date", "name", "amount", "no.", "#", "held", "attended")
_STRONG_RELATIONS = frozenset({"same_row", "nearby_right", "below_label", "same_column_below"})

_COOC_BOOST = {
    "same_row_both_in_patch": 0.35,
    "cross_patch_row_or_right": 0.30,
    "below_label": 0.25,
    "same_column_table": 0.20,
}


@dataclass
class OCRRow:
    row_id: int
    box_indices: list[int]
    boxes: list[dict[str, Any]]
    text: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    mean_conf: float
    token_count: int
    has_question_token: bool
    has_answer_type_token: bool
    row_type: str = "dense_text_row"
    cy: float = 0.0
    cx: float = 0.0


@dataclass
class LabelValueCandidate:
    label_idx: int
    value_idx: int
    label_row_id: int
    value_row_id: int
    relation: str
    value_score: float
    boost_key: str
    boost_value: float


def _box_cy(box: list[list[int]]) -> float:
    return sum(p[1] for p in box) / len(box)


def _box_cx(box: list[list[int]]) -> float:
    return sum(p[0] for p in box) / len(box)


def _box_height(box: list[list[int]]) -> float:
    ys = [p[1] for p in box]
    return max(ys) - min(ys)


def _box_bounds(box: list[list[int]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return min(xs), min(ys), max(xs), max(ys)


def _is_label_box(text: str, question: str) -> bool:
    t = text.lower()
    qtoks = question_tokens(question)
    if any(tok in t for tok in qtoks):
        return True
    return any(h in t for h in _LABEL_HINTS)


def _is_value_box(text: str, question: str) -> bool:
    if not text.strip():
        return False
    if answer_type_score(question, text) > 0:
        return True
    if _NUMERIC_RE.search(text):
        return 1 <= len(tokenize(text)) <= 8
    return len(tokenize(text)) <= 5


def cluster_ocr_rows(ocr_boxes: list[dict[str, Any]], question: str) -> list[OCRRow]:
    """Adaptive y-clustering of OCR boxes into rows."""
    usable = [(i, b) for i, b in enumerate(ocr_boxes) if b.get("text", "").strip()]
    if not usable:
        return []

    heights = [_box_height(b["box"]) for _, b in usable]
    median_h = statistics.median(heights) if heights else 20.0
    thresh = 0.6 * median_h

    usable.sort(key=lambda x: _box_cy(x[1]["box"]))
    groups: list[list[tuple[int, dict[str, Any]]]] = []
    cur: list[tuple[int, dict[str, Any]]] = []
    cur_cy: float | None = None

    for idx, box in usable:
        cy = _box_cy(box["box"])
        if cur_cy is None or abs(cy - cur_cy) <= thresh:
            cur.append((idx, box))
            cur_cy = statistics.mean(_box_cy(b["box"]) for _, b in cur)
        else:
            groups.append(cur)
            cur = [(idx, box)]
            cur_cy = cy
    if cur:
        groups.append(cur)

    rows: list[OCRRow] = []
    qtoks = set(question_tokens(question))
    for rid, group in enumerate(groups):
        texts = [b.get("text", "") for _, b in group]
        joined = " ".join(texts)
        confs = [float(b.get("confidence", 0.0)) for _, b in group]
        x0 = min(_box_bounds(b["box"])[0] for _, b in group)
        y0 = min(_box_bounds(b["box"])[1] for _, b in group)
        x1 = max(_box_bounds(b["box"])[2] for _, b in group)
        y1 = max(_box_bounds(b["box"])[3] for _, b in group)
        norm = normalize_text(joined)
        has_q = any(t in norm for t in qtoks) or ":" in joined
        has_at = answer_type_score(question, joined) > 0 or bool(_NUMERIC_RE.search(joined))
        n_tok = len(tokenize(joined))
        n_boxes = len(group)

        if has_q and n_tok <= 12:
            rtype = "question_row"
        elif has_at and not has_q:
            rtype = "value_row"
        elif n_boxes >= 3 or (n_boxes >= 2 and _NUMERIC_RE.search(joined)):
            rtype = "table_like_row"
        else:
            rtype = "dense_text_row"

        rows.append(OCRRow(
            row_id=rid,
            box_indices=[i for i, _ in group],
            boxes=[b for _, b in group],
            text=joined,
            x_min=x0, y_min=y0, x_max=x1, y_max=y1,
            mean_conf=statistics.mean(confs) if confs else 0.0,
            token_count=n_tok,
            has_question_token=has_q,
            has_answer_type_token=has_at,
            row_type=rtype,
            cy=statistics.mean(_box_cy(b["box"]) for _, b in group),
            cx=statistics.mean(_box_cx(b["box"]) for _, b in group),
        ))
    return rows


def _row_by_box(rows: list[OCRRow]) -> dict[int, OCRRow]:
    out: dict[int, OCRRow] = {}
    for row in rows:
        for bi in row.box_indices:
            out[bi] = row
    return out


def _relation_between(
    label_idx: int,
    value_idx: int,
    ocr_boxes: list[dict[str, Any]],
    rows: list[OCRRow],
    row_map: dict[int, OCRRow],
) -> str | None:
    lb = ocr_boxes[label_idx]["box"]
    vb = ocr_boxes[value_idx]["box"]
    lcx, lcy = _box_cx(lb), _box_cy(lb)
    vcx, vcy = _box_cx(vb), _box_cy(vb)
    lr = row_map.get(label_idx)
    vr = row_map.get(value_idx)
    if lr is None or vr is None:
        return None

    if lr.row_id == vr.row_id and vcx > lcx and (vcx - lcx) < 280:
        return "nearby_right"
    if lr.row_id == vr.row_id:
        return "same_row"
    if vr.row_id == lr.row_id + 1 and abs(vcx - lcx) < 55:
        return "below_label"
    if vr.row_id > lr.row_id and abs(vcx - lcx) < 22:
        return "same_column_below"
    if vr.row_id == lr.row_id + 1:
        return "next_row"
    return None


def _table_alignment_score(lr: OCRRow, vr: OCRRow) -> float:
    if lr.row_type == "table_like_row" and vr.row_type == "table_like_row":
        if abs(lr.cx - vr.cx) < 30:
            return 0.20
        return 0.10
    return 0.0


def _value_score_for_pair(
    label_idx: int,
    value_idx: int,
    relation: str,
    question: str,
    ocr_boxes: list[dict[str, Any]],
    rows: list[OCRRow],
    row_map: dict[int, OCRRow],
) -> float:
    vtext = ocr_boxes[value_idx].get("text", "")
    vconf = float(ocr_boxes[value_idx].get("confidence", 0.0))
    at = answer_type_score(question, vtext)
    rel_w = {
        "same_row": 0.25,
        "nearby_right": 0.22,
        "below_label": 0.20,
        "same_column_below": 0.18,
        "next_row": 0.08,
    }.get(relation, 0.0)

    lr, vr = row_map[label_idx], row_map[value_idx]
    align = _table_alignment_score(lr, vr)
    score = at + rel_w + align + 0.10 * min(1.0, vconf)

    if relation not in _STRONG_RELATIONS:
        score -= 0.25
    return score


def enumerate_label_value_candidates(
    ocr_boxes: list[dict[str, Any]],
    rows: list[OCRRow],
    question: str,
) -> list[LabelValueCandidate]:
    row_map = _row_by_box(rows)
    cands: list[LabelValueCandidate] = []
    labels = [i for i, b in enumerate(ocr_boxes) if _is_label_box(b.get("text", ""), question)]
    if not labels:
        labels = [i for i, row in row_map.items() if row.has_question_token][:8]

    for li in labels:
        for vi, b in enumerate(ocr_boxes):
            if vi == li or not _is_value_box(b.get("text", ""), question):
                continue
            rel = _relation_between(li, vi, ocr_boxes, rows, row_map)
            if rel is None:
                continue
            vs = _value_score_for_pair(li, vi, rel, question, ocr_boxes, rows, row_map)
            boost_key = ""
            boost_val = 0.0
            if rel == "same_row":
                boost_key, boost_val = "same_row_both_in_patch", _COOC_BOOST["same_row_both_in_patch"]
            elif rel in ("same_row", "nearby_right"):
                boost_key, boost_val = "cross_patch_row_or_right", _COOC_BOOST["cross_patch_row_or_right"]
            elif rel == "below_label":
                boost_key, boost_val = "below_label", _COOC_BOOST["below_label"]
            elif rel == "same_column_below":
                boost_key, boost_val = "same_column_table", _COOC_BOOST["same_column_table"]
            cands.append(LabelValueCandidate(
                li, vi, row_map[li].row_id, row_map[vi].row_id,
                rel, vs, boost_key, boost_val,
            ))
    cands.sort(key=lambda c: c.value_score + c.boost_value, reverse=True)
    return cands


def _box_in_patch(box: list[list[int]], patch: Patch) -> bool:
    cx, cy = _box_cx(box), _box_cy(box)
    return patch.x <= cx < patch.x + patch.w and patch.y <= cy < patch.y + patch.h


def _cooccurrence_boost_in_patch(
    patch: Patch,
    cands: list[LabelValueCandidate],
    ocr_boxes: list[dict[str, Any]],
) -> tuple[float, LabelValueCandidate | None]:
    best = 0.0
    best_c: LabelValueCandidate | None = None
    for c in cands[:32]:
        l_in = _box_in_patch(ocr_boxes[c.label_idx]["box"], patch)
        v_in = _box_in_patch(ocr_boxes[c.value_idx]["box"], patch)
        if l_in and v_in and c.relation == "same_row":
            if c.boost_value > best:
                best, best_c = c.boost_value, c
        elif l_in and v_in and c.relation == "below_label":
            b = _COOC_BOOST["below_label"]
            if b > best:
                best, best_c = b, c
        elif l_in and v_in and c.relation == "same_column_below":
            b = _COOC_BOOST["same_column_table"]
            if b > best:
                best, best_c = b, c
    return best, best_c


def _cross_patch_boost(
    patch1: Patch,
    patch2: Patch,
    cands: list[LabelValueCandidate],
    ocr_boxes: list[dict[str, Any]],
) -> tuple[float, LabelValueCandidate | None]:
    for c in cands[:32]:
        l_in = _box_in_patch(ocr_boxes[c.label_idx]["box"], patch1)
        v_in = _box_in_patch(ocr_boxes[c.value_idx]["box"], patch2)
        if not l_in or not v_in:
            continue
        if c.relation in ("same_row", "nearby_right"):
            return _COOC_BOOST["cross_patch_row_or_right"], c
        if c.relation == "below_label":
            return _COOC_BOOST["below_label"], c
        if c.relation == "same_column_below":
            return _COOC_BOOST["same_column_table"], c
    return 0.0, None


def score_table_patch(
    image: Image.Image,
    patch: Patch,
    question: str,
    ocr_boxes: list[dict[str, Any]],
    rows: list[OCRRow],
    cands: list[LabelValueCandidate],
) -> tuple[float, dict[str, Any]]:
    text = patch_ocr_text(patch, ocr_boxes)
    q_ov = question_token_overlap_score(patch, ocr_boxes, question)
    qa = score_patch_question_aware(image, patch, ocr_boxes, question)
    lexical = 0.55 * late_interaction_score(question, text) + 0.45 * bm25_score(question, text)

    pair_part = 0.0
    best_c: LabelValueCandidate | None = None
    for c in cands[:24]:
        l_in = _box_in_patch(ocr_boxes[c.label_idx]["box"], patch)
        v_in = _box_in_patch(ocr_boxes[c.value_idx]["box"], patch)
        if l_in or v_in:
            part = c.value_score + (c.boost_value if (l_in and v_in) else 0.0)
            if part > pair_part:
                pair_part, best_c = part, c

    boost, boost_c = _cooccurrence_boost_in_patch(patch, cands, ocr_boxes)
    if boost_c and (best_c is None or boost > pair_part * 0.5):
        best_c = boost_c

    score = 0.40 * pair_part + 0.30 * qa + 0.20 * lexical + boost
    if q_ov < 0.30:
        layout_excess = max(0.0, score - lexical - pair_part - boost)
        score = lexical + pair_part + boost + min(layout_excess, 0.15)

    detail = {
        "table_pair_score": score,
        "question_overlap": q_ov,
        "cooccurrence_boost": boost,
        "best_relation": best_c.relation if best_c else "",
        "best_label_text": ocr_boxes[best_c.label_idx].get("text", "")[:80] if best_c else "",
        "best_value_text": ocr_boxes[best_c.value_idx].get("text", "")[:80] if best_c else "",
    }
    return score, detail


def select_table_pair_patches(
    image: Image.Image,
    pool: list[Patch],
    k: int,
    question: str,
    ocr_boxes: list[dict[str, Any]],
    *,
    swap_margin: float = 0.05,
) -> tuple[list[Patch], list[float], list[float], dict[str, Any]]:
    """Q-BOPS slot-1 anchor; slot-2 from table/label-value if beats Q slot-2 + margin."""
    meta: dict[str, Any] = {"swap_margin": swap_margin}
    if not pool or not ocr_boxes:
        return [], [], [], meta

    rows = cluster_ocr_rows(ocr_boxes, question)
    cands = enumerate_label_value_candidates(ocr_boxes, rows, question)
    meta["row_count"] = len(rows)
    meta["label_value_candidates"] = len(cands)
    if cands:
        meta["best_candidate_relation"] = cands[0].relation
        meta["best_candidate_score"] = cands[0].value_score

    qa_scores = [score_patch_question_aware(image, p, ocr_boxes, question) for p in pool]
    first = max(zip(pool, qa_scores), key=lambda x: (x[1], -x[0].index))[0]
    qa_mmr = mmr_select(pool, qa_scores, min(k, len(pool)), lambda_=0.5)
    slot2_q = qa_mmr[1] if len(qa_mmr) > 1 else None
    q2_score = qa_scores[pool.index(slot2_q)] if slot2_q else 0.0

    table_scored: list[tuple[Patch, float, dict[str, Any]]] = []
    for p in pool:
        ts, detail = score_table_patch(image, p, question, ocr_boxes, rows, cands)
        table_scored.append((p, ts, detail))
    table_scored.sort(key=lambda x: (x[1], -x[0].index), reverse=True)

    best_table: Patch | None = None
    best_table_score = float("-inf")
    best_detail: dict[str, Any] = {}
    best_cross: LabelValueCandidate | None = None
    for p, ts, detail in table_scored:
        if p.index == first.index:
            continue
        if patch_iou(p, first) >= 0.35:
            continue
        cross, cross_c = _cross_patch_boost(first, p, cands, ocr_boxes)
        total = ts + cross
        if total > best_table_score:
            best_table_score, best_table, best_detail, best_cross = total, p, detail, cross_c

    selected = [first]
    scores = [qa_scores[pool.index(first)]]
    meta["slot1_source"] = "q_bops_anchor"

    if k >= 2:
        if best_table is not None and best_table_score > q2_score + swap_margin:
            selected.append(best_table)
            scores.append(best_table_score)
            meta["slot2_source"] = "table_pair"
            meta["slot2_swap"] = True
            meta.update(best_detail)
            if best_cross:
                meta["cross_patch_relation"] = best_cross.relation
                meta["cross_patch_boost"] = best_cross.boost_value
        elif slot2_q is not None:
            selected.append(slot2_q)
            scores.append(q2_score)
            meta["slot2_source"] = "q_bops_slot2"
            meta["slot2_swap"] = False

    while len(selected) < k:
        for p, ts, _ in table_scored:
            if p.index not in {s.index for s in selected}:
                selected.append(p)
                scores.append(ts)
                break
        else:
            break

    all_scores = [ts for _, ts, _ in table_scored]
    meta["matched_row_text"] = next(
        (r.text[:120] for r in rows if cands and r.row_id == cands[0].label_row_id), "",
    )
    return selected[:k], scores[:k], all_scores, meta
