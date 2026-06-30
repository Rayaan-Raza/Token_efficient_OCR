"""Aggregate budget and efficiency metrics."""

from __future__ import annotations

from typing import Any

import pandas as pd


def filter_valid_budget(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if not r.get("invalid_budget", False)]


def aggregate_ocr_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
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
