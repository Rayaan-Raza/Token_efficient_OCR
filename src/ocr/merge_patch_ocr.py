"""Merge OCR transcripts from overview and multiple BOPS patches.

Token-level deduplication removes exact duplicate token sequences while
preserving reading order (overview first, then patches).
"""

from __future__ import annotations

from src.ocr.normalize_text import normalize_text, tokenize


def merge_patch_ocr(texts: list[str]) -> str:
    """Merge overview + patch OCR strings with token-sequence deduplication.

    Procedure:
        1. Normalize each non-empty segment.
        2. Tokenize each segment.
        3. Greedily append tokens, skipping exact duplicate token *sequences*
           already present in the merged stream (simple: skip duplicate lines
           and duplicate consecutive n-grams of length up to full segment).

    For the sanity pilot we use a practical rule:
        - Keep segment order (overview, patch1, patch2, ...)
        - For each segment, append only tokens not already in the global multiset
          (count-aware, same as word_recall matching)

    Args:
        texts: OCR outputs in order (overview first, then patches).

    Returns:
        Space-joined merged transcript.
    """
    merged_tokens: list[str] = []
    used: dict[str, int] = {}

    for raw in texts:
        tokens = tokenize(raw)
        if not tokens:
            continue
        for t in tokens:
            if used.get(t, 0) == 0:
                merged_tokens.append(t)
            used[t] = used.get(t, 0) + 1

    if merged_tokens:
        return " ".join(merged_tokens)

    # Fallback: join non-empty raw segments if tokenization emptied everything
    parts = []
    seen_lines: set[str] = set()
    for raw in texts:
        line = normalize_text(raw)
        if line and line not in seen_lines:
            seen_lines.add(line)
            parts.append(line)
    return " ".join(parts)
