"""RAVEN-Select: reader-aware answer-output selection among budgeted DocVQA readers.

The production method is the frozen OCR-grounded shortest-answer rule stamped in
``method_spec``. Learned selectors and feature-group variants are comparators /
ablations only. Gold answers are used only for training labels inside OOF folds
— never as inference features.
"""

from __future__ import annotations

from src.answer_selection.features import FEATURE_GROUPS, build_output_features, feature_keys
from src.answer_selection.method_spec import (
    METHOD_VERSION,
    PRODUCTION_FLAGS,
    PRODUCTION_READERS,
    TIE_BREAK_ORDER,
    method_stamp,
)
from src.answer_selection.train import evaluate_selector

__all__ = [
    "FEATURE_GROUPS",
    "METHOD_VERSION",
    "PRODUCTION_FLAGS",
    "PRODUCTION_READERS",
    "TIE_BREAK_ORDER",
    "build_output_features",
    "evaluate_selector",
    "feature_keys",
    "method_stamp",
]
