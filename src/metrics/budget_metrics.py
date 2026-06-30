"""Aggregate experiment metrics with budget-fairness filtering.

Rows with ``invalid_budget=True`` must be excluded before computing paper
tables or plots. This module provides helpers used by analysis scripts.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def filter_valid_budget(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove result rows that failed budget compliance checks.

    Args:
        rows: Per-sample eval results with optional ``invalid_budget`` field.

    Returns:
        Subset where ``invalid_budget`` is false or absent.
    """
    return [r for r in rows if not r.get("invalid_budget", False)]


def aggregate_ocr_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute mean CER/WER/word recall over valid-budget OCR results.

    Args:
        rows: OCR eval rows (e.g. from ``ocr_metrics.csv``).

    Returns:
        Dict with ``cer_mean``, ``wer_mean``, ``word_recall_mean``, ``n``,
        and ``invalid_excluded`` count. Empty dict if no valid rows.
    """
    valid = filter_valid_budget(rows)
    if not valid:
        return {}
    df = pd.DataFrame(valid)
    return {
        "cer_mean": float(df["cer"].mean()) if "cer" in df else 0.0,
        "wer_mean": float(df["wer"].mean()) if "wer" in df else 0.0,
        "word_recall_mean": float(df["word_recall"].mean()) if "word_recall" in df else 0.0,
        "n": len(valid),
        "invalid_excluded": len(rows) - len(valid),
    }
