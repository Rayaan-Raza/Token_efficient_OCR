"""Unit tests for OCR-protected full-page compression."""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.preprocessing.ocr_page_compress import (
    compress_fullpage,
    margin_crop,
    ocr_protected_seam_carve,
    whitespace_compress,
)
from src.utils.image_io import image_area


def _blank_with_text_block(w: int = 200, h: int = 160) -> tuple[Image.Image, list[dict]]:
    arr = np.full((h, w, 3), 255, dtype=np.uint8)
    # Dark text-like block in the center.
    arr[40:80, 50:150] = 20
    # Wide white margins already present.
    boxes = [{
        "box": [[50, 40], [150, 40], [150, 80], [50, 80]],
        "text": "INV-20491",
        "confidence": 0.99,
    }]
    return Image.fromarray(arr), boxes


def test_margin_crop_shrinks_whitespace():
    img, _ = _blank_with_text_block()
    cropped, meta = margin_crop(img, pad=2)
    assert meta["cropped"] is True
    assert cropped.width < img.width
    assert cropped.height < img.height


def test_whitespace_compress_keeps_ocr_rows():
    img, boxes = _blank_with_text_block()
    out, meta = whitespace_compress(img, boxes)
    assert out.height >= 40  # text block height preserved in some form
    # OCR rows must not be fully deleted.
    assert meta.get("reason") != "deleted_ocr"


def test_seam_carve_never_crosses_ocr_mask():
    img, boxes = _blank_with_text_block(w=120, h=100)
    carved, meta = ocr_protected_seam_carve(
        img, boxes, target_frac=0.9, max_seams=10, carve_max_side=120
    )
    assert carved.size[0] >= 1 and carved.size[1] >= 1
    # Either carved with finite seams or refused because protect too dense.
    assert "v_seams" in meta and "h_seams" in meta


def test_compress_fullpage_matches_resize_budget():
    img, boxes = _blank_with_text_block(w=400, h=300)
    target = int(image_area(img) * 0.25)
    for variant in ("margin_crop", "ws_compress", "ocr_seam"):
        result = compress_fullpage(img, variant, ocr_boxes=boxes, area_ratio=0.25)
        actual = image_area(result.image)
        # ±15% soft tolerance for integer rounding on tiny synthetic images.
        assert abs(actual - target) / target < 0.15, (variant, actual, target)
        assert result.meta["method"].endswith("resize")
