"""Document VQA metrics: Exact Match and ANLS.

Used for DocVQA evaluation (Phase 9–10). Ground truth may be a list of
acceptable answer strings; metrics take the best score across references.

ANLS uses normalized Levenshtein distance with a 0.5 threshold (DocVQA standard).
"""

from __future__ import annotations

import re
import string


def _normalize(s: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s)


def exact_match(prediction: str, ground_truths: list[str]) -> float:
    """Return 1.0 if prediction exactly matches any normalized reference.

    Args:
        prediction: Model answer string.
        ground_truths: List of acceptable answers from the dataset.

    Returns:
        ``1.0`` or ``0.0``.
    """
    pred = _normalize(prediction)
    for gt in ground_truths:
        if pred == _normalize(gt):
            return 1.0
    return 0.0


def _levenshtein(a: str, b: str) -> int:
    """Classic dynamic-programming edit distance."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def anls(prediction: str, ground_truths: list[str], threshold: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity (best over references).

    Score is ``1 - normalized_distance`` if distance ratio < ``threshold``,
    else ``0``.

    Args:
        prediction: Model answer.
        ground_truths: Acceptable reference answers.
        threshold: ANLS cutoff (default 0.5 per DocVQA).

    Returns:
        Score in [0, 1].
    """
    pred = _normalize(prediction)
    if not ground_truths:
        return 0.0
    best = 0.0
    for gt in ground_truths:
        g = _normalize(gt)
        if not g and not pred:
            return 1.0
        if not g or not pred:
            continue
        dist = _levenshtein(pred, g)
        nl = dist / max(len(pred), len(g))
        score = 1.0 - nl if nl < threshold else 0.0
        best = max(best, score)
    return best
