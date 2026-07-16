"""Equal-cost ensemble baselines over cached multi-path VLM predictions.

All baselines use the same three path outputs as RAVEN-post (resize, BM25, LER)
and therefore have equal VLM-call cost. They do not use gold labels.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Sequence

from src.routing import normalize_pred


def _digits(s: str) -> str:
    return "".join(re.findall(r"\d+", s or ""))


def pick_majority(preds: dict[str, str], *, default: str = "resize") -> str:
    norms = {m: normalize_pred(p) for m, p in preds.items()}
    nonempty = {m: n for m, n in norms.items() if n}
    if not nonempty:
        return default
    counts = Counter(nonempty.values())
    best_text, _ = counts.most_common(1)[0]
    # Prefer default among ties for majority text.
    for m, n in nonempty.items():
        if n == best_text and m == default:
            return m
    for m, n in nonempty.items():
        if n == best_text:
            return m
    return default


def pick_resize_on_disagree(preds: dict[str, str], *, default: str = "resize") -> str:
    norms = {m: normalize_pred(p) for m, p in preds.items()}
    vals = [n for n in norms.values() if n]
    if len(set(vals)) <= 1:
        return pick_majority(preds, default=default)
    return default


def pick_prefer_method_on_disagree(
    preds: dict[str, str],
    prefer: str,
    *,
    default: str = "resize",
) -> str:
    norms = {m: normalize_pred(p) for m, p in preds.items()}
    vals = [n for n in norms.values() if n]
    if len(set(vals)) <= 1:
        return pick_majority(preds, default=default)
    if prefer in preds and norms.get(prefer):
        return prefer
    return default


def pick_shortest_nonempty(preds: dict[str, str], *, default: str = "resize") -> str:
    cands = [(m, normalize_pred(p)) for m, p in preds.items() if normalize_pred(p)]
    if not cands:
        return default
    # Length first; then production tie order resize → BM25 → LER-BOPS.
    # Keep the order local to avoid a routing → answer_selection import cycle.
    tie = {"resize": 0, "bm25": 1, "ler_bops": 2}
    cands.sort(key=lambda x: (len(x[1]), tie.get(x[0], 99)))
    return cands[0][0]


def pick_longest_nonempty(preds: dict[str, str], *, default: str = "resize") -> str:
    cands = [(m, normalize_pred(p)) for m, p in preds.items() if normalize_pred(p)]
    if not cands:
        return default
    cands.sort(key=lambda x: (-len(x[1]), 0 if x[0] == default else 1))
    return cands[0][0]


def pick_max_digit_overlap(
    preds: dict[str, str],
    question: str,
    *,
    default: str = "resize",
) -> str:
    q_digits = _digits(question)
    best_m, best_score = default, -1
    for m, p in preds.items():
        n = normalize_pred(p)
        if not n:
            continue
        pd = _digits(n)
        if not q_digits:
            score = len(pd)  # prefer numeric-looking answers
        else:
            score = sum(1 for c in q_digits if c in pd)
        if score > best_score or (score == best_score and m == default):
            best_score, best_m = score, m
    return best_m


def pick_uniform_random(
    methods: Sequence[str],
    rng,
) -> str:
    return str(rng.choice(list(methods)))


ENSEMBLE_NAMES = [
    "majority_vote",
    "resize_on_disagree",
    "bm25_on_disagree",
    "ler_on_disagree",
    "shortest_nonempty",
    "longest_nonempty",
    "max_digit_overlap",
    "uniform_random",
]


def choose_ensemble(
    name: str,
    preds: dict[str, str],
    *,
    question: str = "",
    methods: Sequence[str] | None = None,
    default: str = "resize",
    rng=None,
) -> str:
    methods = list(methods or preds.keys())
    if name == "majority_vote":
        return pick_majority(preds, default=default)
    if name == "resize_on_disagree":
        return pick_resize_on_disagree(preds, default=default)
    if name == "bm25_on_disagree":
        return pick_prefer_method_on_disagree(preds, "bm25", default=default)
    if name == "ler_on_disagree":
        return pick_prefer_method_on_disagree(preds, "ler_bops", default=default)
    if name == "shortest_nonempty":
        return pick_shortest_nonempty(preds, default=default)
    if name == "longest_nonempty":
        return pick_longest_nonempty(preds, default=default)
    if name == "max_digit_overlap":
        return pick_max_digit_overlap(preds, question, default=default)
    if name == "uniform_random":
        if rng is None:
            import numpy as np

            rng = np.random.RandomState(0)
        return pick_uniform_random(methods, rng)
    raise ValueError(f"Unknown ensemble: {name}")
