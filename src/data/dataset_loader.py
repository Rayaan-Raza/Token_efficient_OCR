"""Unified JSONL manifest loading for OCR and VLM runners.

All experiments iterate manifests through :func:`iter_manifest` or load them
fully with :func:`load_manifest`. Each row follows the schema in the research
plan (image_id, dataset, split, image_path, ocr_gt_text, question, answer, ...).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def load_manifest(manifest_path: str | Path) -> list[dict]:
    """Load entire manifest into memory.

    Args:
        manifest_path: Path to ``.jsonl`` file.

    Returns:
        List of manifest record dicts.
    """
    rows = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def iter_manifest(manifest_path: str | Path) -> Iterator[dict]:
    """Stream manifest rows one at a time (memory-efficient).

    Args:
        manifest_path: Path to ``.jsonl`` file.

    Yields:
        One manifest record dict per non-empty line.
    """
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
