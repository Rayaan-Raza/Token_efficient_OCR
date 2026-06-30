"""OCR metrics: CER, WER, word recall (canonical v1)."""

from __future__ import annotations

import jiwer

from src.ocr.normalize_text import normalize_text, tokenize


def cer(prediction: str, ground_truth: str) -> float:
    p, g = normalize_text(prediction), normalize_text(ground_truth)
    if not g:
        return 0.0 if not p else 1.0
    return float(jiwer.cer(g, p))


def wer(prediction: str, ground_truth: str) -> float:
    p, g = normalize_text(prediction), normalize_text(ground_truth)
    if not g:
        return 0.0 if not p else 1.0
    return float(jiwer.wer(g, p))


def word_recall(prediction: str, ground_truth: str) -> float:
    """matched normalized GT tokens / total GT tokens (exact, count-aware)."""
    gt_tokens = tokenize(ground_truth)
    if not gt_tokens:
        return 1.0 if not tokenize(prediction) else 0.0
    pred_tokens = tokenize(prediction)
    pred_counts: dict[str, int] = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    matched = 0
    for t in gt_tokens:
        if pred_counts.get(t, 0) > 0:
            matched += 1
            pred_counts[t] -= 1
    return matched / len(gt_tokens)
