"""Uniform grid candidate patch generation."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass
class Patch:
    x: int
    y: int
    w: int
    h: int
    index: int

    def clamp(self, img_w: int, img_h: int) -> "Patch":
        x = max(0, min(self.x, img_w - 1))
        y = max(0, min(self.y, img_h - 1))
        w = max(1, min(self.w, img_w - x))
        h = max(1, min(self.h, img_h - y))
        return Patch(x, y, w, h, self.index)

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h, "index": self.index}


def generate_grid_patches(image: Image.Image, patch_size: int, stride: int) -> list[Patch]:
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
    p = patch.clamp(image.width, image.height)
    return image.crop((p.x, p.y, p.x + p.w, p.y + p.h))
