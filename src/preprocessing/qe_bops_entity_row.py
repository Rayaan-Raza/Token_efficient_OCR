"""QE-BOPS entity-aware row-specific label matching (final K=2 structural selector)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from PIL import Image

from src.features.patch_features import answer_type_score
from src.ocr.normalize_text import normalize_text, tokenize
from src.preprocessing.mmr_select import mmr_select, patch_iou
from src.preprocessing.patch_grid import Patch
from src.preprocessing.patch_scoring_qa import (
    patch_ocr_text,
    question_tokens,
    score_patch_question_aware,
)
from src.preprocessing.qe_bops_node_pair import late_interaction_score
from src.preprocessing.qe_bops_table_pair import OCRRow, cluster_ocr_rows


_NUMERIC_RE = re.compile(r"\d[\d,./%-]*")
_INIT_RE = re.compile(r"\b[A-Z](?:\.\s*[A-Z])+\.?|\b[A-Z]\.\s*[A-Z]\.|\b[A-Z]\.\s*[A-Z][a-z]+")
_QUOTED_RE = re.compile(r"['\"]([^'\"]+)['\"]")
_CAPS_PHRASE_RE = re.compile(r"\b[A-Z][A-Za-z]*(?:\.\s*[A-Z][A-Za-z]*)+(?:\s+[A-Z][a-z]+)?|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")

_QUESTION_ONLY_STOP = frozenset({
    "what", "who", "whom", "when", "where", "how", "much", "many", "which",
    "the", "is", "was", "were", "are", "did", "does", "have", "has", "had",
    "with", "that", "this",
})

_ENTITY_STOP = _QUESTION_ONLY_STOP | frozenset({
    "date", "amount", "total", "final", "weight", "age", "name", "address",
    "value", "number", "time", "session", "document", "title", "type",
})

_FIELD_HINTS = frozenset({
    "final", "wt", "weight", "initial", "age", "options", "held", "pay", "paid",
    "amount", "total", "vein", "diet", "title", "time", "name", "brands", "brand",
    "lifestyle", "apparel", "session", "answers", "question", "options", "no",
})

def _looks_like_field_phrase(phrase: str) -> bool:
    low = phrase.lower()
    return any(h in low for h in _FIELD_HINTS) or bool(re.search(r"\b(wt|no\.?|#)\b", low, re.I))


_ENTITY_ANCHORS = frozenset({
    "subject", "patient", "employee", "party", "child", "group", "director",
    "name", "person", "student", "member", "brand", "company",
})

_ENTITY_BOOST = {
    "entity_field_value_same_row": 0.45,
    "entity_value_same_row": 0.35,
    "entity_field_column": 0.30,
    "entity_below_value": 0.25,
}


@dataclass
class QuestionParse:
    entity_tokens: list[str]
    field_tokens: list[str]
    answer_type_hint: str


def _norm_field_tokens(raw: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for t in raw:
        nt = normalize_text(t) if not re.match(r"^\d", t) else t
        if not nt or nt in _QUESTION_ONLY_STOP or len(str(nt)) <= 1:
            continue
        if nt not in seen:
            seen.add(nt)
            out.append(nt)
    return out


def _norm_entity_tokens(raw: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for t in raw:
        nt = normalize_text(t) if not re.match(r"^\d", t) else t
        if not nt or nt in _ENTITY_STOP or len(str(nt)) <= 1:
            continue
        if nt not in seen:
            seen.add(nt)
            out.append(nt)
    return out


def extract_entity_field_tokens(question: str) -> QuestionParse:
    """Split question into row-entity vs field tokens (rule-based, no NER)."""
    q_lower = question.lower()
    qtoks = question_tokens(question)
    entity_raw: list[str] = []
    field_raw: list[str] = list(qtoks)

    for m in _INIT_RE.finditer(question):
        entity_raw.append(normalize_text(m.group(0).replace(" ", "")))
        entity_raw.append(normalize_text(m.group(0)))

    for m in _QUOTED_RE.finditer(question):
        entity_raw.extend(tokenize(m.group(1)))

    for m in _CAPS_PHRASE_RE.finditer(question):
        phrase = m.group(0)
        if len(phrase) >= 2 and not _looks_like_field_phrase(phrase):
            entity_raw.append(normalize_text(phrase))
            entity_raw.extend(tokenize(phrase))

    for t in qtoks:
        if t in _FIELD_HINTS:
            field_raw.append(t)

    for anchor in _ENTITY_ANCHORS:
        if anchor in q_lower:
            idx = q_lower.find(anchor)
            after = question[idx + len(anchor):].strip(" ?.:,")
            entity_raw.extend(tokenize(after)[:3])

    for prep in (" of ", " for ", " by ", " to ", " from "):
        if prep in q_lower:
            tail = question[q_lower.rfind(prep) + len(prep):]
            entity_raw.extend(tokenize(tail))

    for m in re.finditer(r"\b[A-Z](?:\.\s*[A-Z])+\.?\s+[A-Z][a-z]+\b", question):
        entity_raw.append(normalize_text(m.group(0)))
        entity_raw.extend(tokenize(m.group(0)))

    if re.search(r"\b\d+\.?\d*\b", question):
        for m in re.finditer(r"\b\d+\.?\d*\b", question):
            field_raw.append(m.group(0))
            if float(m.group(0).replace(",", "")) > 10 or "." in m.group(0):
                entity_raw.append(m.group(0))

    entity_tokens = _norm_entity_tokens(entity_raw)
    field_tokens = _norm_field_tokens([
        t for t in field_raw
        if t not in entity_tokens and t not in _QUESTION_ONLY_STOP
    ])

    at_hint = "number"
    if any(w in q_lower for w in ("date", "when", "year", "month", "day", "time")):
        at_hint = "date"
    elif any(w in q_lower for w in ("amount", "price", "cost", "fee", "pay", "paid")):
        at_hint = "currency"
    elif any(w in q_lower for w in ("name", "who", "title")):
        at_hint = "name"
    elif "percent" in q_lower or "%" in q_lower:
        at_hint = "percent"

    return QuestionParse(entity_tokens, field_tokens, at_hint)


def _overlap_score(text: str, tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    norm = normalize_text(text)
    hits = sum(1 for t in tokens if t in norm)
    return hits / len(tokens)


def _row_has_value(row: OCRRow, question: str) -> bool:
    if answer_type_score(question, row.text) > 0:
        return True
    return bool(_NUMERIC_RE.search(row.text))


def score_entity_row(
    row: OCRRow,
    parse: QuestionParse,
    question: str,
    *,
    relation_score: float = 0.0,
) -> float:
    entity = _overlap_score(row.text, parse.entity_tokens)
    field = _overlap_score(row.text, parse.field_tokens)
    value = 1.0 if _row_has_value(row, question) else 0.0
    if parse.answer_type_hint == "name" and any(t in row.text.lower() for t in parse.entity_tokens):
        value = max(value, 0.5)

    score = 0.40 * entity + 0.25 * field + 0.20 * value + 0.15 * relation_score
    if parse.entity_tokens and entity == 0.0:
        score -= 0.40
    if parse.field_tokens and field > 0 and parse.entity_tokens and entity == 0.0:
        score -= 0.30
    return score


def _best_row_relation(
    label_row: OCRRow,
    value_row: OCRRow,
) -> tuple[str, float]:
    if label_row.row_id == value_row.row_id:
        return "same_row", 1.0
    if value_row.row_id == label_row.row_id + 1 and abs(value_row.cx - label_row.cx) < 55:
        return "below_label", 0.85
    if label_row.row_id == value_row.row_id and value_row.cx > label_row.cx:
        return "nearby_right", 0.90
    if abs(value_row.cx - label_row.cx) < 22 and value_row.row_id > label_row.row_id:
        return "same_column_below", 0.75
    return "next_row", 0.35


def _entity_boost(
    parse: QuestionParse,
    label_row: OCRRow,
    value_row: OCRRow | None,
    question: str,
) -> tuple[float, str]:
    entity = _overlap_score(label_row.text, parse.entity_tokens)
    field = _overlap_score(label_row.text, parse.field_tokens)
    if parse.entity_tokens and entity == 0.0:
        return 0.0, ""

    if value_row is None:
        if entity > 0 and field > 0 and _row_has_value(label_row, question):
            return _ENTITY_BOOST["entity_field_value_same_row"], "entity_field_value_same_row"
        if entity > 0 and field > 0:
            return -0.30, "generic_field_no_entity_value"
        return 0.0, ""

    rel, rel_s = _best_row_relation(label_row, value_row)
    has_value = _row_has_value(value_row, question) or _row_has_value(label_row, question)
    if entity > 0 and field > 0 and has_value and rel == "same_row":
        return _ENTITY_BOOST["entity_field_value_same_row"], "entity_field_value_same_row"
    if entity > 0 and has_value and rel in ("same_row", "nearby_right"):
        return _ENTITY_BOOST["entity_value_same_row"], "entity_value_same_row"
    if entity > 0 and field > 0 and rel == "same_column_below":
        return _ENTITY_BOOST["entity_field_column"], "entity_field_column"
    if entity > 0 and has_value and rel == "below_label":
        return _ENTITY_BOOST["entity_below_value"], "entity_below_value"
    if field > 0 and entity == 0.0:
        return -0.30, "generic_field_no_entity"
    return 0.15 * rel_s, rel


def _row_in_patch(row: OCRRow, patch: Patch) -> bool:
    return (
        patch.x <= row.cx < patch.x + patch.w
        and patch.y <= row.cy < patch.y + patch.h
    )


def score_entity_patch(
    image: Image.Image,
    patch: Patch,
    rows: list[OCRRow],
    parse: QuestionParse,
    question: str,
    ocr_boxes: list[dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    row_scores: list[tuple[OCRRow, float]] = []
    for row in rows:
        rs = score_entity_row(row, parse, question)
        row_scores.append((row, rs))
    row_scores.sort(key=lambda x: x[1], reverse=True)

    best_boost = 0.0
    boost_key = ""
    best_label_row: OCRRow | None = None
    best_value_row: OCRRow | None = None
    best_rel = ""

    for label_row, _ in row_scores[:12]:
        if not _row_in_patch(label_row, patch):
            continue
        for value_row in rows:
            if label_row.row_id != value_row.row_id and not _row_in_patch(value_row, patch):
                if value_row.row_id not in (label_row.row_id, label_row.row_id + 1):
                    continue
            b, key = _entity_boost(parse, label_row, value_row, question)
            if b > best_boost:
                best_boost, boost_key = b, key
                best_label_row, best_value_row = label_row, value_row
                best_rel, _ = _best_row_relation(label_row, value_row)

        solo_b, solo_key = _entity_boost(parse, label_row, None, question)
        if solo_b > best_boost:
            best_boost, boost_key = solo_b, solo_key
            best_label_row, best_value_row = label_row, None
            best_rel = "same_row"

    top_row_score = max((rs for r, rs in row_scores if _row_in_patch(r, patch)), default=0.0)
    text = patch_ocr_text(patch, ocr_boxes)
    lexical = late_interaction_score(question, text)
    qa = score_patch_question_aware(image, patch, ocr_boxes, question)

    score = 0.35 * top_row_score + 0.25 * qa + 0.20 * lexical + best_boost
    if parse.entity_tokens and _overlap_score(text, parse.entity_tokens) == 0.0 and best_boost <= 0:
        score -= 0.25

    detail = {
        "entity_row_score": score,
        "top_row_score": top_row_score,
        "boost_applied": best_boost,
        "boost_key": boost_key,
        "relation_type": best_rel,
        "selected_row_text": (best_label_row.text[:120] if best_label_row else ""),
        "selected_value_text": (best_value_row.text[:80] if best_value_row else ""),
        "entity_tokens": ",".join(parse.entity_tokens),
        "field_tokens": ",".join(parse.field_tokens),
        "answer_type_hint": parse.answer_type_hint,
    }
    return score, detail


def _patch_has_entity_and_value(
    patch: Patch,
    parse: QuestionParse,
    rows: list[OCRRow],
    question: str,
) -> bool:
    for row in rows:
        if not _row_in_patch(row, patch):
            continue
        if parse.entity_tokens and _overlap_score(row.text, parse.entity_tokens) == 0:
            continue
        if _row_has_value(row, question) or _overlap_score(row.text, parse.field_tokens) > 0:
            return True
    return False


def select_entity_row_patches(
    image: Image.Image,
    pool: list[Patch],
    k: int,
    question: str,
    ocr_boxes: list[dict[str, Any]],
    *,
    swap_margin: float = 0.05,
    slot1_replace_min: float = 0.85,
) -> tuple[list[Patch], list[float], list[float], dict[str, Any]]:
    """Q-BOPS slot-1 anchor; slot-2 from entity-row evidence if beats Q slot-2 + margin."""
    meta: dict[str, Any] = {"swap_margin": swap_margin}
    if not pool or not ocr_boxes:
        return [], [], [], meta

    parse = extract_entity_field_tokens(question)
    meta["entity_tokens"] = parse.entity_tokens
    meta["field_tokens"] = parse.field_tokens
    meta["answer_type_hint"] = parse.answer_type_hint

    rows = cluster_ocr_rows(ocr_boxes, question)
    meta["row_count"] = len(rows)

    qa_scores = [score_patch_question_aware(image, p, ocr_boxes, question) for p in pool]
    first = max(zip(pool, qa_scores), key=lambda x: (x[1], -x[0].index))[0]
    qa_mmr = mmr_select(pool, qa_scores, min(k, len(pool)), lambda_=0.5)
    slot2_q = qa_mmr[1] if len(qa_mmr) > 1 else None
    q2_score = qa_scores[pool.index(slot2_q)] if slot2_q else 0.0

    entity_scored: list[tuple[Patch, float, dict[str, Any]]] = []
    for p in pool:
        es, detail = score_entity_patch(image, p, rows, parse, question, ocr_boxes)
        entity_scored.append((p, es, detail))
    entity_scored.sort(key=lambda x: (x[1], -x[0].index), reverse=True)

    best_entity: Patch | None = None
    best_entity_score = float("-inf")
    best_detail: dict[str, Any] = {}
    for p, es, detail in entity_scored:
        if p.index == first.index:
            continue
        if patch_iou(p, first) >= 0.35:
            continue
        if es > best_entity_score:
            best_entity_score, best_entity, best_detail = es, p, detail

    slot1 = first
    slot1_score = qa_scores[pool.index(first)]
    meta["slot1_source"] = "q_bops_anchor"

    if entity_scored and entity_scored[0][0].index != first.index:
        top_p, top_es, top_detail = entity_scored[0]
        if (
            top_es >= slot1_replace_min
            and parse.entity_tokens
            and _patch_has_entity_and_value(top_p, parse, rows, question)
            and top_es > slot1_score + 0.12
        ):
            slot1 = top_p
            slot1_score = top_es
            meta["slot1_source"] = "entity_row_high_confidence"
            meta.update(top_detail)

    selected = [slot1]
    scores = [slot1_score]

    if k >= 2:
        if best_entity is not None and best_entity_score > q2_score + swap_margin:
            selected.append(best_entity)
            scores.append(best_entity_score)
            meta["slot2_source"] = "entity_row"
            meta["slot2_swap"] = True
            meta.update(best_detail)
        elif slot2_q is not None and slot2_q.index != slot1.index:
            selected.append(slot2_q)
            scores.append(q2_score)
            meta["slot2_source"] = "q_bops_slot2"
            meta["slot2_swap"] = False
        elif best_entity is not None and best_entity.index != slot1.index:
            selected.append(best_entity)
            scores.append(best_entity_score)
            meta["slot2_source"] = "entity_row_fallback"
            meta["slot2_swap"] = False

    while len(selected) < k:
        for p, es, _ in entity_scored:
            if p.index not in {s.index for s in selected}:
                selected.append(p)
                scores.append(es)
                break
        else:
            break

    all_scores = [es for _, es, _ in entity_scored]
    return selected[:k], scores[:k], all_scores, meta
