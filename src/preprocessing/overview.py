"""Low-resolution global overview generation for BOPS.

The overview preserves layout and global context at a fixed pixel budget while
high-resolution patches capture small text. This is Step 2 of the BOPS pipeline.
"""

from __future__ import annotations

from PIL import Image

from src.utils.image_io import image_area


def generate_overview(image: Image.Image, target_pixels: int) -> tuple[Image.Image, dict]:
    """Downscale an image so its area is at most ``target_pixels``.

    If the image is already smaller than the budget, returns a copy unchanged.

    Args:
        image: Full-resolution source image.
        target_pixels: Maximum width × height for the overview.

    Returns:
        Tuple of (overview image, metadata with scale and pixel counts).
    """
    orig_area = image_area(image)
    if orig_area <= target_pixels:
        return image.copy(), {"overview_scale": 1.0, "target_pixels": target_pixels}
    scale = (target_pixels / orig_area) ** 0.5
    w = max(1, int(image.width * scale))
    h = max(1, int(image.height * scale))
    overview = image.resize((w, h), Image.Resampling.LANCZOS)
    return overview, {"overview_scale": scale, "target_pixels": target_pixels, "overview_pixels": w * h}
