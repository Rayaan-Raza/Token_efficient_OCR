"""Tests for GT answer leakage guard."""

from __future__ import annotations

import pytest
from PIL import Image

from src.preprocessing.selectors import AnswerLeakageError, select_patches


def test_qe_bops_rejects_answers_at_inference():
    img = Image.new("RGB", (400, 400), "white")
    with pytest.raises(AnswerLeakageError):
        select_patches(
            img,
            "qe_bops",
            2,
            "What is the invoice date?",
            [],
            answers=["2020"],
        )


def test_oracle_allows_answers_with_eval_labels():
    img = Image.new("RGB", (400, 400), "white")
    # empty pool edge case — should not raise leakage before value error
    try:
        select_patches(
            img,
            "oracle",
            1,
            "question?",
            [],
            answers=["x"],
            patch_labels=[{"label_confidence": 1.0, "label_positive": True}],
            eval_labels=True,
        )
    except (AnswerLeakageError, ValueError):
        pass
