"""Post-hoc patch-selection diagnostics for VLM DocVQA evaluation.

Ground-truth answers are used only for post-hoc diagnostics and are never used
for patch selection, preprocessing, prompting, or model inference.
"""

from __future__ import annotations

import json
import re
from typing import Any

from PIL import Image

from src.ocr.run_ocr import run_ocr_on_image, run_ocr_with_boxes
from src.preprocessing.patch_grid import Patch
from src.preprocessing.patch_scoring import text_coverage_score
from src.utils.ocr_cache import (
    load_cached_ocr_boxes,
    load_cached_patch_ocr,
    save_cached_ocr_boxes,
    save_cached_patch_ocr,
)


def _normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace for substring checks."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _answer_in_text(answers: list[str], text: str) -> bool:
    """Return True if any normalized answer appears in normalized OCR text."""
    norm_text = _normalize_text(text)
    if not norm_text:
        return False
    for ans in answers:
        norm_ans = _normalize_text(ans)
        if norm_ans and norm_ans in norm_text:
            return True
    return False


def _patch_from_dict(d: dict[str, Any]) -> Patch:
    return Patch(
        x=int(d["x"]),
        y=int(d["y"]),
        w=int(d["w"]),
        h=int(d["h"]),
        index=int(d.get("index", 0)),
    )


def _boxes_overlap_patch(patch: Patch, box: list) -> bool:
    from src.preprocessing.patch_scoring import _intersection_area

    return _intersection_area(patch, box) > 0


def get_full_image_ocr_boxes(image: Image.Image, image_id: str) -> list[dict[str, Any]]:
    """Return OCR boxes for full image, using disk cache when available."""
    cached = load_cached_ocr_boxes(image_id)
    if cached is not None:
        return cached
    import tempfile

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(tmp.name)
    boxes = run_ocr_with_boxes(tmp.name)
    save_cached_ocr_boxes(image_id, boxes)
    return boxes


def get_patch_ocr_texts(
    image: Image.Image,
    image_id: str,
    method: str,
    num_patches: int,
    patch_images: list[Image.Image],
) -> list[str]:
    """OCR each selected patch crop, with optional disk cache."""
    cached = load_cached_patch_ocr(image_id, method, num_patches)
    if cached is not None:
        return list(cached.get("patch_texts", []))

    texts: list[str] = []
    for patch_img in patch_images:
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        patch_img.save(tmp.name)
        texts.append(run_ocr_on_image(tmp.name))

    save_cached_patch_ocr(
        image_id,
        method,
        num_patches,
        {"patch_texts": texts},
    )
    return texts


def compute_patch_diagnostics(
    image: Image.Image,
    image_id: str,
    method: str,
    num_patches: int,
    bops_result: dict[str, Any],
    answers: list[str],
) -> dict[str, Any]:
    """Compute post-hoc diagnostics after BOPS preprocessing and VLM inference.

    Args:
        image: Full-resolution source image.
        image_id: Sample identifier for OCR caching.
        method: VLM eval method (``bops``, ``random``, ``uniform``, ``overview_only``).
        num_patches: Configured patch budget.
        bops_result: Output dict from :func:`run_bops`.
        answers: Ground-truth answer strings (post-hoc only).

    Returns:
        Dict of diagnostic fields for CSV logging.
    """
    meta = bops_result.get("meta", {})
    patch_coords = meta.get("patches", [])
    patch_images = bops_result.get("patches", [])
    patch_objs = [_patch_from_dict(p) for p in patch_coords]

    ocr_boxes = get_full_image_ocr_boxes(image, image_id)
    full_text = " ".join(b.get("text", "") for b in ocr_boxes)

    if method == "overview_only" or not patch_images:
        patch_texts: list[str] = []
    else:
        patch_texts = get_patch_ocr_texts(
            image, image_id, method, num_patches, patch_images
        )

    selected_patch_text = " ".join(patch_texts)
    num_boxes_selected = sum(
        1
        for b in ocr_boxes
        if any(_boxes_overlap_patch(p, b["box"]) for p in patch_objs)
    )
    if patch_objs:
        coverages = [text_coverage_score(p, ocr_boxes) for p in patch_objs]
        selected_coverage = float(sum(coverages) / len(coverages))
    else:
        selected_coverage = 0.0

    coords_out = [[p.x, p.y, p.w, p.h] for p in patch_objs]

    return {
        "answer_in_selected_patch_ocr": _answer_in_text(answers, selected_patch_text),
        "answer_in_full_image_ocr": _answer_in_text(answers, full_text),
        "num_ocr_boxes_selected": num_boxes_selected,
        "selected_text_box_coverage": round(selected_coverage, 4),
        "mean_patch_score": meta.get("mean_patch_score", 0.0),
        "selected_patch_coords": json.dumps(coords_out),
    }
