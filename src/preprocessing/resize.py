"""Resize images to target area ratio."""

from __future__ import annotations

import math

from PIL import Image

from src.utils.budget_check import check_pixel_budget, merge_budget_fields
from src.utils.image_io import image_area


def resize_to_area_ratio(image: Image.Image, area_ratio: float) -> tuple[Image.Image, dict]:
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
