"""Unit tests for DocVQA QA metrics (:mod:`src.vlm.qa_metrics`).

Covers Exact Match (normalized string equality) and ANLS (average normalized
Levenshtein similarity) used in VLM evaluation.
"""

from src.vlm.qa_metrics import anls, exact_match


def test_exact_match():
    """Exact match is case-insensitive; different strings score 0."""
    assert exact_match("March 12 2024", ["12 March 2024"]) == 0.0
    assert exact_match("hello", ["hello"]) == 1.0


def test_anls_perfect():
    """Identical answer and reference yield ANLS of 1.0."""
    assert anls("hello", ["hello"]) == 1.0


def test_anls_partial():
    """Near-miss answers yield ANLS in [0, 1]."""
    score = anls("helo", ["hello"])
    assert 0.0 <= score <= 1.0
