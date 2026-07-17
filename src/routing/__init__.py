"""Reader-aware routing for RAVEN-BOPS budgeted Document VQA.

The router chooses, per question, which answering path to trust:
``resize`` (full page), ``bm25`` (lexical patch retrieval), ``ler_bops``
(learned patch retrieval), ``q_bops`` (heuristic patch retrieval), and the
``candidate`` answer-extraction path.

All router features are leakage-safe: they are derived from model predictions,
retrieval signals, and the question -- never from gold answers. Gold answers are
attached only when computing training labels and evaluation metrics.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Any, Sequence

_PUNCT = str.maketrans("", "", string.punctuation)
_NUM_RE = re.compile(r"\d")
_DATE_RE = re.compile(r"\b(19|20)\d{2}\b|\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b")


def normalize_pred(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace (matches qa_metrics)."""
    s = (s or "").lower().strip()
    s = s.translate(_PUNCT)
    return re.sub(r"\s+", " ", s)


def predict_question_type(question: str) -> str:
    q = (question or "").lower()
    if any(w in q for w in ["how many", "how much", "number of", "total", "count"]):
        return "count"
    if any(w in q for w in ["percent", "%", "rate"]):
        return "percent"
    if any(w in q for w in ["when", "date", "year", "day", "month"]):
        return "date"
    if any(w in q for w in ["who", "name of", "whom", "author", "signed"]):
        return "person"
    if any(w in q for w in ["where", "address", "location", "city", "country"]):
        return "location"
    if q.startswith("what is the") or q.startswith("what are"):
        return "value"
    return "phrase"


_QTYPES = ["count", "percent", "date", "person", "location", "value", "phrase"]


def _pred_text_features(pred: str) -> dict[str, float]:
    p = pred or ""
    toks = normalize_pred(p).split()
    return {
        "len_chars": float(len(p)),
        "len_tokens": float(len(toks)),
        "is_empty": 1.0 if not toks else 0.0,
        "has_digit": 1.0 if _NUM_RE.search(p) else 0.0,
        "digit_frac": float(sum(c.isdigit() for c in p) / max(len(p), 1)),
        "has_date": 1.0 if _DATE_RE.search(p) else 0.0,
        "is_single_token": 1.0 if len(toks) == 1 else 0.0,
    }


@dataclass
class RouterExample:
    image_id: str
    question: str
    features: dict[str, float]
    method_anls: dict[str, float]
    method_em: dict[str, float]
    method_pred: dict[str, str]


def build_router_features(
    image_id: str,
    question: str,
    method_pred: dict[str, str],
    methods: Sequence[str],
    *,
    default_method: str = "resize",
) -> dict[str, float]:
    """Construct leakage-safe router features from predictions + question.

    Features:
        - per-method prediction text properties
        - consensus / agreement structure across methods
        - agreement of each method with the default (resize) path
        - question-type one-hot
    """
    feats: dict[str, float] = {}
    norm = {m: normalize_pred(method_pred.get(m, "")) for m in methods}

    for m in methods:
        for k, v in _pred_text_features(method_pred.get(m, "")).items():
            feats[f"{m}__{k}"] = v

    # Consensus structure.
    counts: dict[str, int] = {}
    for m in methods:
        if norm[m]:
            counts[norm[m]] = counts.get(norm[m], 0) + 1
    if counts:
        majority_text, majority_n = max(counts.items(), key=lambda kv: kv[1])
    else:
        majority_text, majority_n = "", 0
    feats["consensus_majority_size"] = float(majority_n)
    feats["consensus_distinct"] = float(len(counts))
    feats["consensus_frac"] = float(majority_n / max(len(methods), 1))

    default_norm = norm.get(default_method, "")
    n_agree_default = sum(1 for m in methods if norm[m] and norm[m] == default_norm)
    feats["agree_with_default_count"] = float(n_agree_default)
    for m in methods:
        feats[f"{m}__agrees_default"] = 1.0 if (norm[m] and norm[m] == default_norm) else 0.0
        feats[f"{m}__in_majority"] = 1.0 if (norm[m] and norm[m] == majority_text) else 0.0

    # Pairwise agreement counts per method.
    for m in methods:
        agree = sum(1 for other in methods if other != m and norm[m] and norm[m] == norm[other])
        feats[f"{m}__agree_count"] = float(agree)

    qtype = predict_question_type(question)
    for qt in _QTYPES:
        feats[f"qtype_{qt}"] = 1.0 if qtype == qt else 0.0
    feats["q_len_tokens"] = float(len(normalize_pred(question).split()))

    return feats


def router_feature_keys(methods: Sequence[str]) -> list[str]:
    """Stable ordered feature-name list for a given method set."""
    dummy_pred = {m: "" for m in methods}
    feats = build_router_features("", "what is the value", dummy_pred, methods)
    return sorted(feats.keys())
