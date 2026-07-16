"""Leakage-safe per-output features for RAVEN-Select.

Each feature is derived from prediction text, question text, cross-route
consensus, OCR presence of the *prediction* (not gold), and optional route
confidence scores. Never use gold answers, ANLS/EM to gold, or answer_in_*_ocr.
"""

from __future__ import annotations

import re
from typing import Sequence

from src.routing import normalize_pred, predict_question_type

_NUM_RE = re.compile(r"\d")
_DATE_RE = re.compile(
    r"\b(19|20)\d{2}\b|\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b|"
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b",
    re.I,
)
_AMOUNT_RE = re.compile(r"[\$€£]|usd|eur|inr|\brs\.?\b", re.I)
_PERCENT_RE = re.compile(r"%|\bpercent\b|\bpercentage\b", re.I)
_PERSON_RE = re.compile(r"\b(mr|mrs|ms|dr|prof)\.?\b|[A-Z][a-z]+\s+[A-Z][a-z]+")
_SENTENCE_RE = re.compile(r"\b(the|is|are|was|were|of|to|and|for|in|on|at)\b", re.I)
_ANSWER_IS_RE = re.compile(r"^\s*(the\s+)?answer\s+is\b", re.I)

_QTYPES = ["count", "percent", "date", "person", "location", "value", "phrase"]
_ROUTES = ["resize", "bm25", "ler_bops"]

FEATURE_GROUPS = {
    "route_id": ["route_"],
    "pred_text": [
        "len_chars", "len_tokens", "is_empty", "is_numeric", "is_date_like",
        "is_amount_like", "is_percent_like", "is_person_like",
        "contains_sentence_words", "starts_with_answer_is", "digit_frac",
        "is_single_token",
    ],
    "answer_type": ["qtype_", "answer_type_match", "pred_type_"],
    "consensus": [
        "agree_count", "agreement_size", "edit_sim_mean", "edit_sim_max",
        "is_substring_of_other", "other_is_substring", "in_majority",
        "is_shortest_nonempty", "is_longest_nonempty", "n_same_normalized",
    ],
    "ocr_presence": ["pred_in_full_ocr", "pred_in_patch_ocr", "ocr_"],
    "route_confidence": ["bm25_", "ler_", "retrieval_"],
}


def _edit_sim(a: str, b: str) -> float:
    """Normalized Levenshtein similarity in [0, 1]."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    n, m = len(a), len(b)
    if n > m:
        a, b, n, m = b, a, m, n
    prev = list(range(n + 1))
    for j, cb in enumerate(b):
        cur = [j + 1]
        for i, ca in enumerate(a):
            ins, delete, sub = prev[i + 1] + 1, cur[i] + 1, prev[i] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    dist = prev[n]
    return 1.0 - dist / max(n, m, 1)


def _pred_type(pred: str) -> str:
    p = pred or ""
    n = normalize_pred(p)
    if not n:
        return "empty"
    if _PERCENT_RE.search(p) or (n.endswith("%") if False else "%" in p):
        return "percent"
    if _AMOUNT_RE.search(p):
        return "amount"
    if _DATE_RE.search(p):
        return "date"
    if _NUM_RE.search(p) and len(n.split()) <= 3:
        return "numeric"
    if _PERSON_RE.search(p) or (len(n.split()) <= 4 and not _NUM_RE.search(p)):
        # weak person heuristic: short non-numeric phrases for who-questions handled via match
        if any(w[0].isupper() for w in (pred or "").split() if w):
            return "person"
    return "phrase"


def _type_match(qtype: str, ptype: str) -> float:
    mapping = {
        "count": {"numeric"},
        "percent": {"percent", "numeric"},
        "date": {"date", "numeric"},
        "person": {"person", "phrase"},
        "location": {"phrase"},
        "value": {"numeric", "amount", "percent", "phrase"},
        "phrase": {"phrase", "person"},
    }
    return 1.0 if ptype in mapping.get(qtype, {ptype}) else 0.0


def build_output_features(
    route: str,
    prediction: str,
    question: str,
    all_preds: dict[str, str],
    *,
    methods: Sequence[str] | None = None,
    ocr_full_text: str = "",
    ocr_patch_text: str = "",
    route_scores: dict[str, float] | None = None,
) -> dict[str, float]:
    """Build leakage-safe features for one (question, route, prediction) row."""
    methods = list(methods or _ROUTES)
    feats: dict[str, float] = {}
    p = prediction or ""
    n = normalize_pred(p)
    toks = n.split()

    # Route id one-hot
    for r in _ROUTES:
        feats[f"route_{r}"] = 1.0 if route == r else 0.0

    # Prediction text properties
    feats["len_chars"] = float(len(p))
    feats["len_tokens"] = float(len(toks))
    feats["is_empty"] = 1.0 if not toks else 0.0
    feats["digit_frac"] = float(sum(c.isdigit() for c in p) / max(len(p), 1))
    feats["is_single_token"] = 1.0 if len(toks) == 1 else 0.0
    alpha_free = not any(c.isalpha() for c in n.replace(" ", ""))
    feats["is_numeric"] = 1.0 if (
        toks and alpha_free and _NUM_RE.search(p) and not _DATE_RE.search(p) and "%" not in p
    ) else 0.0
    feats["is_date_like"] = 1.0 if _DATE_RE.search(p) else 0.0
    feats["is_amount_like"] = 1.0 if _AMOUNT_RE.search(p) else 0.0
    feats["is_percent_like"] = 1.0 if _PERCENT_RE.search(p) or "%" in p else 0.0
    feats["is_person_like"] = 1.0 if _PERSON_RE.search(p) else 0.0
    feats["contains_sentence_words"] = 1.0 if _SENTENCE_RE.search(p) and len(toks) >= 4 else 0.0
    feats["starts_with_answer_is"] = 1.0 if _ANSWER_IS_RE.search(p) else 0.0

    ptype = _pred_type(p)
    for t in ["empty", "numeric", "percent", "amount", "date", "person", "phrase"]:
        feats[f"pred_type_{t}"] = 1.0 if ptype == t else 0.0

    qtype = predict_question_type(question)
    for qt in _QTYPES:
        feats[f"qtype_{qt}"] = 1.0 if qtype == qt else 0.0
    feats["answer_type_match"] = _type_match(qtype, ptype)

    # Consensus / relative length among sibling predictions
    norms = {m: normalize_pred(all_preds.get(m, "")) for m in methods}
    nonempty = [(m, norms[m]) for m in methods if norms[m]]
    counts: dict[str, int] = {}
    for _, nn in nonempty:
        counts[nn] = counts.get(nn, 0) + 1
    majority_n = max(counts.values()) if counts else 0
    majority_text = max(counts, key=counts.get) if counts else ""
    feats["agreement_size"] = float(majority_n)
    feats["n_same_normalized"] = float(counts.get(n, 0) if n else 0)
    feats["in_majority"] = 1.0 if (n and n == majority_text and majority_n >= 2) else 0.0
    feats["agree_count"] = float(
        sum(1 for m in methods if m != route and norms.get(m) and norms[m] == n and n)
    )

    others = [norms[m] for m in methods if m != route and norms[m]]
    if n and others:
        sims = [_edit_sim(n, o) for o in others]
        feats["edit_sim_mean"] = float(sum(sims) / len(sims))
        feats["edit_sim_max"] = float(max(sims))
        feats["is_substring_of_other"] = 1.0 if any(n in o and n != o for o in others) else 0.0
        feats["other_is_substring"] = 1.0 if any(o in n and n != o for o in others) else 0.0
    else:
        feats["edit_sim_mean"] = 0.0
        feats["edit_sim_max"] = 0.0
        feats["is_substring_of_other"] = 0.0
        feats["other_is_substring"] = 0.0

    if nonempty:
        shortest = min(nonempty, key=lambda x: (len(x[1]), 0 if x[0] == "resize" else 1))
        longest = max(nonempty, key=lambda x: (len(x[1]), 0 if x[0] == "resize" else 1))
        feats["is_shortest_nonempty"] = 1.0 if route == shortest[0] else 0.0
        feats["is_longest_nonempty"] = 1.0 if route == longest[0] else 0.0
    else:
        feats["is_shortest_nonempty"] = 0.0
        feats["is_longest_nonempty"] = 0.0

    # OCR presence of prediction (never gold)
    full_n = normalize_pred(ocr_full_text)
    patch_n = normalize_pred(ocr_patch_text)
    feats["pred_in_full_ocr"] = 1.0 if (n and n in full_n) else 0.0
    feats["pred_in_patch_ocr"] = 1.0 if (n and n in patch_n) else 0.0
    feats["ocr_full_chars"] = float(len(full_n))
    feats["ocr_patch_chars"] = float(len(patch_n))

    # Route confidence / retrieval (from pre-feature parquet)
    rs = route_scores or {}
    for k in (
        "bm25_top1", "bm25_top2", "bm25_gap", "bm25_mean", "bm25_std",
        "ler_top1", "ler_top2", "ler_gap", "ler_mean", "ler_std",
        "retrieval_jaccard", "retrieval_overlap_count",
        "ocr_n_boxes", "ocr_mean_conf", "ocr_total_chars", "ocr_total_tokens",
    ):
        feats[k] = float(rs.get(k, 0.0))

    return feats


def feature_keys(methods: Sequence[str] | None = None) -> list[str]:
    """Stable ordered feature names."""
    methods = list(methods or _ROUTES)
    dummy = {m: "x" for m in methods}
    feats = build_output_features("resize", "12", "what is the date", dummy, methods=methods)
    return sorted(feats.keys())


def keys_for_groups(groups: Sequence[str], all_keys: Sequence[str] | None = None) -> list[str]:
    """Return feature keys belonging to named FEATURE_GROUPS."""
    keys = list(all_keys or feature_keys())
    prefixes: list[str] = []
    for g in groups:
        prefixes.extend(FEATURE_GROUPS.get(g, [g]))
    return [k for k in keys if any(p in k or k.startswith(p) or k == p for p in prefixes)]
