"""Candidate patch grid generation and cropping.

Produces a sliding-window grid of fixed-size patches over the source image.
Patches are clamped to image bounds before cropping. Used as the candidate set
for OCR-guided selection, random baselines, and uniform tiling.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass
class Patch:
    """Axis-aligned rectangular region in image coordinates.

    Attributes:
        x: Left edge (pixels).
        y: Top edge (pixels).
        w: Width (pixels).
        h: Height (pixels).
        index: Sequential index in the candidate grid.
    """

    x: int
    y: int
    w: int
    h: int
    index: int

    def clamp(self, img_w: int, img_h: int) -> "Patch":
        """Clip patch rectangle to fit inside an ``img_w`` × ``img_h`` image.

        Args:
            img_w: Image width.
            img_h: Image height.

        Returns:
            New :class:`Patch` with coordinates clamped to valid bounds.
        """
        x = max(0, min(self.x, img_w - 1))
        y = max(0, min(self.y, img_h - 1))
        w = max(1, min(self.w, img_w - x))
        h = max(1, min(self.h, img_h - y))
        return Patch(x, y, w, h, self.index)

    def as_dict(self) -> dict:
        """Serialize patch coordinates for JSON/metadata output."""
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h, "index": self.index}


def generate_grid_patches(image: Image.Image, patch_size: int, stride: int) -> list[Patch]:
    """Generate overlapping grid patches across the full image.

    Args:
        image: Source PIL image.
        patch_size: Width and height of each patch (square).
        stride: Step between patch origins (smaller = more overlap).

    Returns:
        List of :class:`Patch` objects, edge-clamped to the image.
    """
    patches = []
    idx = 0
    for y in range(0, image.height, stride):
        for x in range(0, image.width, stride):
            p = Patch(x, y, patch_size, patch_size, idx).clamp(image.width, image.height)
            if p.w > 0 and p.h > 0:
                patches.append(p)
                idx += 1
    return patches


def crop_patch(image: Image.Image, patch: Patch) -> Image.Image:
    """Extract a sub-image for the given patch region.

    Args:
        image: Source PIL image.
        patch: Region to crop (clamped before extraction).

    Returns:
        Cropped PIL image.
    """
    p = patch.clamp(image.width, image.height)
    return image.crop((p.x, p.y, p.x + p.w, p.y + p.h))
