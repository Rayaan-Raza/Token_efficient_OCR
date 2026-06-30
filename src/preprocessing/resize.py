"""Area-ratio resize baseline for budget-normalized preprocessing.

Implements the simplest baseline in the BOPS study: scale image dimensions so
total pixel count matches a target fraction of the original area (e.g. 50%, 25%).
Budget compliance is checked via :mod:`src.utils.budget_check` (±3% tolerance).

Example::

    from src.preprocessing.resize import resize_to_area_ratio

    resized, meta = resize_to_area_ratio(image, area_ratio=0.5)
    assert meta["invalid_budget"] is False
"""

from __future__ import annotations

import math

from PIL import Image

from src.utils.budget_check import check_pixel_budget, merge_budget_fields
from src.utils.image_io import image_area


def resize_to_area_ratio(image: Image.Image, area_ratio: float) -> tuple[Image.Image, dict]:
    """Resize an image to approximately ``area_ratio`` of its original pixel area.

    Uses LANCZOS resampling and preserves aspect ratio by applying a uniform
    scale factor of ``sqrt(area_ratio)`` to width and height.

    Args:
        image: Source PIL image.
        area_ratio: Target fraction of original area (e.g. ``0.5`` for 50%).

    Returns:
        Tuple of (resized image, metadata dict including budget fields).
    """
    orig_area = image_area(image)
    target_area = orig_area * area_ratio
    scale = math.sqrt(area_ratio)
    new_w = max(1, int(image.width * scale))
    new_h = max(1, int(image.height * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    meta = {"method": "resize", "area_ratio": area_ratio, "original_area": orig_area}
    budget = check_pixel_budget(image_area(resized), int(target_area))
    merge_budget_fields(meta, budget)
    return resized, meta
