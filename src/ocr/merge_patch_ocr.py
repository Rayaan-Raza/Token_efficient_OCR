"""Merge OCR transcripts from multiple patches into one string.

When BOPS emits several high-resolution patches, each may be OCR'd separately.
This module deduplicates identical line strings while preserving reading order
(first occurrence wins).
"""

from __future__ import annotations


def merge_patch_ocr(texts: list[str]) -> str:
    """Concatenate unique non-empty patch OCR strings in order.

    Args:
        texts: OCR output per patch (in reading/importance order).

    Returns:
        Space-joined merged transcript.
    """
    seen = set()
    parts = []
    for t in texts:
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            parts.append(t)
    return " ".join(parts)
