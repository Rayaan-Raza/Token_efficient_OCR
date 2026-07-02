"""Unit tests for canonical OCR metrics (:mod:`src.ocr.ocr_metrics`).

Covers CER, WER, and word recall v1 (normalized exact token matching).
These definitions are shared across all OCR evaluation scripts.
"""

from src.ocr.ocr_metrics import cer, wer, word_recall

import pytest


def test_cer_wer_perfect_match():
    """Identical prediction and ground truth yield zero CER and WER."""
    assert cer("hello world", "hello world") == 0.0
    assert wer("hello world", "hello world") == 0.0


def test_word_recall_perfect():
    """Full token overlap yields word recall of 1.0."""
    assert word_recall("hello world", "hello world") == 1.0


def test_word_recall_partial():
    """Partial token overlap yields proportional word recall."""
    assert word_recall("hello", "hello world") == 0.5


def test_word_precision_and_f1():
    from src.ocr.ocr_metrics import word_f1, word_precision

    assert word_precision("hello world", "hello world") == 1.0
    assert word_precision("hello foo bar", "hello world") == pytest.approx(1 / 3)
    assert word_f1("hello", "hello world") == pytest.approx(2 * 0.5 * 1.0 / (0.5 + 1.0))


def test_token_stats():
    from src.ocr.ocr_metrics import duplicate_token_ratio, predicted_token_count

    assert predicted_token_count("hello world") == 2
    assert duplicate_token_ratio("a a b") == pytest.approx(1 / 3)
