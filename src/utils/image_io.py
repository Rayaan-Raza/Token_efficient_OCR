"""Image load/save helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from PIL import Image

from src.utils.paths import ensure_dir


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_image(image: Image.Image, path: str | Path, **kwargs: Any) -> Path:
    out = Path(path)
    ensure_dir(out.parent)
    image.save(out, **kwargs)
    return out


def image_area(image: Image.Image) -> int:
    return image.width * image.height


def write_metadata_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    out = Path(path)
    ensure_dir(out.parent)
    if not rows:
        out.write_text("", encoding="utf-8")
        return out
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return out
