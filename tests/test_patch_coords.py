"""Unit tests for patch grid geometry (:mod:`src.preprocessing.patch_grid`).

Ensures patch clamping keeps crops inside image bounds before BOPS selection.
"""

from PIL import Image

from src.preprocessing.patch_grid import Patch, crop_patch


def test_patch_inside_bounds():
    """Clamped patch crops must not exceed image width or height."""
    img = Image.new("RGB", (100, 100))
    p = Patch(10, 10, 30, 30, 0).clamp(100, 100)
    crop = crop_patch(img, p)
    assert crop.size == (30, 30)
    assert p.x + p.w <= 100
    assert p.y + p.h <= 100
