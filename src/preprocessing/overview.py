"""Generate low-resolution overview of full image."""

from __future__ import annotations

from PIL import Image

from src.utils.image_io import image_area


def generate_overview(image: Image.Image, target_pixels: int) -> tuple[Image.Image, dict]:
    orig_area = image_area(image)
    if orig_area <= target_pixels:
        return image.copy(), {"overview_scale": 1.0, "target_pixels": target_pixels}
    scale = (target_pixels / orig_area) ** 0.5
    w = max(1, int(image.width * scale))
    h = max(1, int(image.height * scale))
    overview = image.resize((w, h), Image.Resampling.LANCZOS)
    return overview, {"overview_scale": scale, "target_pixels": target_pixels, "overview_pixels": w * h}
