"""OCR evaluation metrics: CER, WER, and word recall (canonical v1).

**Word recall (v1)** — single definition used everywhere in this repo::

    word_recall = matched_gt_tokens / total_gt_tokens

Matching uses normalized exact tokens (count-aware multiset). See
:func:`word_recall` for details. Do not reimplement elsewhere.

CER/WER are computed via ``jiwer`` on normalized strings.
"""

from __future__ import annotations

import jiwer

from src.ocr.normalize_text import normalize_text, tokenize


def cer(prediction: str, ground_truth: str) -> float:
    """Character Error Rate between prediction and ground truth.

    Args:
        prediction: OCR output string.
        ground_truth: Reference text.

    Returns:
        CER in [0, ∞); 0 is perfect. Empty GT with non-empty pred returns 1.0.
    """
    p, g = normalize_text(prediction), normalize_text(ground_truth)
    if not g:
        return 0.0 if not p else 1.0
    return float(jiwer.cer(g, p))


def wer(prediction: str, ground_truth: str) -> float:
    """Word Error Rate between prediction and ground truth.

    Args:
        prediction: OCR output string.
        ground_truth: Reference text.

    Returns:
        WER in [0, ∞); 0 is perfect.
    """
    p, g = normalize_text(prediction), normalize_text(ground_truth)
    if not g:
        return 0.0 if not p else 1.0
    return float(jiwer.wer(g, p))


def word_recall(prediction: str, ground_truth: str) -> float:
    """Canonical v1 word recall: matched GT tokens / total GT tokens.

    Procedure:
        1. Normalize and tokenize both strings.
        2. Each GT token matches at most one identical prediction token
           (count-aware).

    Args:
        prediction: OCR output string.
        ground_truth: Reference text.

    Returns:
        Recall in [0, 1]. Empty GT with empty pred → 1.0; empty GT with pred → 0.0.
    """
    gt_tokens = tokenize(ground_truth)
    if not gt_tokens:
        return 1.0 if not tokenize(prediction) else 0.0
    matched, _, _ = _token_match_counts(prediction, ground_truth)
    return matched / len(gt_tokens)


def _token_match_counts(prediction: str, ground_truth: str) -> tuple[int, int, int]:
    """Return (matched, gt_count, pred_count) with count-aware exact token matching."""
    gt_tokens = tokenize(ground_truth)
    pred_tokens = tokenize(prediction)
    pred_counts: dict[str, int] = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    matched = 0
    for t in gt_tokens:
        if pred_counts.get(t, 0) > 0:
            matched += 1
            pred_counts[t] -= 1
    return matched, len(gt_tokens), len(pred_tokens)


def word_precision(prediction: str, ground_truth: str) -> float:
    """Matched prediction tokens / total prediction tokens (count-aware)."""
    _, _, pred_count = _token_match_counts(prediction, ground_truth)
    if pred_count == 0:
        return 1.0 if not tokenize(ground_truth) else 0.0
    matched, _, _ = _token_match_counts(prediction, ground_truth)
    return matched / pred_count


def word_f1(prediction: str, ground_truth: str) -> float:
    """Harmonic mean of word precision and word recall."""
    p = word_precision(prediction, ground_truth)
    r = word_recall(prediction, ground_truth)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def predicted_token_count(prediction: str) -> int:
    """Number of normalized tokens in the OCR prediction."""
    return len(tokenize(prediction))


def duplicate_token_ratio(prediction: str) -> float:
    """Fraction of prediction tokens that are repeated (1 - unique/total)."""
    tokens = tokenize(prediction)
    if not tokens:
        return 0.0
    return 1.0 - len(set(tokens)) / len(tokens)
