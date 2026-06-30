"""OCR-guided patch scoring for BOPS (Step 5 of the pipeline).

Combines four signals to rank candidate patches:
    - text_coverage: fraction of OCR box area overlapped by the patch
    - text_confidence: mean OCR confidence for boxes in the patch
    - edge_density: gradient magnitude (text-like structure)
    - entropy: grayscale histogram entropy (information content)

Default weights favor OCR signals (0.4 + 0.3) with smaller weight on
image-only cues (0.15 + 0.15). Used by :mod:`src.preprocessing.bops` before NMS.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from src.preprocessing.patch_grid import Patch


def _box_area(box: list) -> float:
    """Compute axis-aligned area of a quadrilateral OCR box."""
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return max(0, max(xs) - min(xs)) * max(0, max(ys) - min(ys))


def _intersection_area(patch: Patch, box: list) -> float:
    """Pixel area of overlap between a patch rectangle and an OCR box."""
    bx0, bx1 = min(p[0] for p in box), max(p[0] for p in box)
    by0, by1 = min(p[1] for p in box), max(p[1] for p in box)
    px0, py0 = patch.x, patch.y
    px1, py1 = patch.x + patch.w, patch.y + patch.h
    ix0, iy0 = max(px0, bx0), max(py0, by0)
    ix1, iy1 = min(px1, bx1), min(py1, by1)
    return max(0, ix1 - ix0) * max(0, iy1 - iy0)


def text_coverage_score(patch: Patch, ocr_boxes: list[dict[str, Any]]) -> float:
    """Fraction of total OCR text box area covered by this patch.

    Args:
        patch: Candidate patch region.
        ocr_boxes: List of dicts with ``box`` (4-point polygon) keys.

    Returns:
        Value in [0, 1]; 0 if no OCR boxes exist.
    """
    total_text = 0.0
    covered = 0.0
    for b in ocr_boxes:
        box = b["box"]
        area = _box_area(box)
        if area <= 0:
            continue
        total_text += area
        covered += _intersection_area(patch, box)
    return covered / total_text if total_text > 0 else 0.0


def text_confidence_score(patch: Patch, ocr_boxes: list[dict[str, Any]]) -> float:
    """Mean OCR confidence for boxes intersecting the patch.

    Args:
        patch: Candidate patch region.
        ocr_boxes: OCR detections with ``box`` and optional ``confidence``.

    Returns:
        Mean confidence in [0, 1], or 0.0 if no overlapping boxes.
    """
    confs = []
    for b in ocr_boxes:
        if _intersection_area(patch, b["box"]) > 0:
            confs.append(b.get("confidence", 0.0))
    return float(np.mean(confs)) if confs else 0.0


def edge_density_score(image: Image.Image, patch: Patch) -> float:
    """Normalized mean gradient magnitude inside the patch (text proxy).

    Args:
        image: Full source image.
        patch: Region to score.

    Returns:
        Scalar in roughly [0, 1].
    """
    crop = image.crop((patch.x, patch.y, patch.x + patch.w, patch.y + patch.h))
    gray = np.array(crop.convert("L"), dtype=np.float32)
    if gray.size == 0:
        return 0.0
    gx = np.abs(np.diff(gray, axis=1)).mean() if gray.shape[1] > 1 else 0.0
    gy = np.abs(np.diff(gray, axis=0)).mean() if gray.shape[0] > 1 else 0.0
    return float((gx + gy) / 2.0 / 255.0)


def entropy_score(image: Image.Image, patch: Patch) -> float:
    """Normalized grayscale entropy inside the patch.

    Args:
        image: Full source image.
        patch: Region to score.

    Returns:
        Entropy scaled by log2(256) ≈ 8.
    """
    crop = image.crop((patch.x, patch.y, patch.x + patch.w, patch.y + patch.h))
    gray = np.array(crop.convert("L"))
    if gray.size == 0:
        return 0.0
    hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    return float(-(hist * np.log2(hist)).sum() / 8.0)


def score_patch(
    image: Image.Image,
    patch: Patch,
    ocr_boxes: list[dict[str, Any]],
    weights: dict[str, float] | None = None,
) -> float:
    """Compute weighted patch importance score.

    Args:
        image: Full source image (for edge/entropy).
        patch: Candidate patch.
        ocr_boxes: OCR detections from the full image.
        weights: Optional override for component weights.

    Returns:
        Combined score (higher = more important for selection).
    """
    w = weights or {
        "text_coverage": 0.4,
        "text_confidence": 0.3,
        "edge_density": 0.15,
        "entropy": 0.15,
    }
    tc = text_coverage_score(patch, ocr_boxes)
    tconf = text_confidence_score(patch, ocr_boxes)
    edge = edge_density_score(image, patch)
    ent = entropy_score(image, patch)
    return (
        w["text_coverage"] * tc
        + w["text_confidence"] * tconf
        + w["edge_density"] * edge
        + w["entropy"] * ent
    )
