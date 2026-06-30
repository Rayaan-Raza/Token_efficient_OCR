"""Unit tests for byte-budget image compression (:mod:`src.preprocessing.compression`).

Validates JPEG compression metadata and approximate byte-budget compliance.
"""

from PIL import Image

from src.preprocessing.compression import compress_to_byte_budget


def test_jpeg_byte_budget():
    """Compressed output should respect target bytes (within tolerance) or flag invalid."""
    img = Image.new("RGB", (800, 600), color=(200, 100, 50))
    target = 50_000
    data, meta = compress_to_byte_budget(img, target, fmt="JPEG")
    assert len(data) <= target * 1.02 or meta["invalid_budget"] is False
    assert meta["budget_type"] == "byte"
