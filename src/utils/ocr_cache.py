"""Disk cache for OCR boxes and patch OCR text during VLM diagnostics.

Caches avoid repeated EasyOCR calls across methods and samples at pilot scale.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.paths import outputs_path


def _boxes_cache_path(image_id: str) -> Path:
    return outputs_path("cache", "ocr_boxes", f"{image_id}.json")


def _patch_ocr_cache_path(image_id: str, method: str, num_patches: int) -> Path:
    return outputs_path("cache", "patch_ocr", f"{image_id}_{method}_k{num_patches}.json")


def load_cached_ocr_boxes(image_id: str) -> list[dict[str, Any]] | None:
    """Load cached full-image OCR boxes for ``image_id``, or None if missing."""
    path = _boxes_cache_path(image_id)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        path.unlink(missing_ok=True)
        return None


def _json_safe_boxes(boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OCR box structures to JSON-serializable Python types."""
    out: list[dict[str, Any]] = []
    for b in boxes:
        box = [[int(x), int(y)] for x, y in b["box"]]
        out.append({
            "box": box,
            "text": b.get("text", ""),
            "confidence": float(b.get("confidence", 0.0)),
        })
    return out


def save_cached_ocr_boxes(image_id: str, boxes: list[dict[str, Any]]) -> Path:
    """Persist full-image OCR boxes to disk cache."""
    path = _boxes_cache_path(image_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_json_safe_boxes(boxes), f, ensure_ascii=False)
    tmp.replace(path)
    return path


def load_cached_patch_ocr(image_id: str, method: str, num_patches: int) -> dict[str, Any] | None:
    """Load cached patch OCR payload, or None if missing."""
    path = _patch_ocr_cache_path(image_id, method, num_patches)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_cached_patch_ocr(
    image_id: str,
    method: str,
    num_patches: int,
    payload: dict[str, Any],
) -> Path:
    """Persist patch OCR text and metadata to disk cache."""
    path = _patch_ocr_cache_path(image_id, method, num_patches)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return path
