"""Merge OCR text from multiple patches (dedupe by reading order)."""

from __future__ import annotations


def merge_patch_ocr(texts: list[str]) -> str:
    seen = set()
    parts = []
    for t in texts:
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            parts.append(t)
    return " ".join(parts)
