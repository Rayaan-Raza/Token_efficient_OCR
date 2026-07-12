"""Fair pool hash must match across fair selectors."""

from __future__ import annotations

from PIL import Image

from src.preprocessing.selectors import assert_same_pool_hash, select_patches


def test_fair_methods_share_pool_hash():
    img = Image.new("RGB", (512, 512), "white")
    q = "What is the total amount?"
    boxes = [{"box": [[10, 10], [100, 10], [100, 30], [10, 30]], "text": "Total Amount", "confidence": 0.9}]
    a = select_patches(img, "bops_fair_pool", 2, q, boxes)
    b = select_patches(img, "qe_bops", 2, q, boxes)
    assert_same_pool_hash(a, b)
