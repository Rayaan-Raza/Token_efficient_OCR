"""Image I/O and experiment metadata CSV helpers.

Provides a consistent RGB loading path for OCR/VLM pipelines and a CSV writer
that unions field names across heterogeneous result rows (e.g. rows with
optional ``error`` or budget fields).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from PIL import Image

from src.utils.paths import ensure_dir


def load_image(path: str | Path) -> Image.Image:
    """Load an image from disk and convert to RGB.

    Args:
        path: Image file path.

    Returns:
        PIL Image in RGB mode (no alpha/palette surprises for OCR).
    """
    return Image.open(path).convert("RGB")


def save_image(image: Image.Image, path: str | Path, **kwargs: Any) -> Path:
    """Save a PIL image, creating parent directories as needed.

    Args:
        image: Source image.
        path: Output file path.
        **kwargs: Extra arguments passed to :meth:`PIL.Image.Image.save`.

    Returns:
        Resolved output path.
    """
    out = Path(path)
    ensure_dir(out.parent)
    image.save(out, **kwargs)
    return out


def image_area(image: Image.Image) -> int:
    """Return pixel count (width × height) for budget calculations.

    Args:
        image: PIL Image.

    Returns:
        Total number of pixels.
    """
    return image.width * image.height


def write_metadata_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write a list of homogeneous or heterogeneous dicts to CSV.

    Field names are the union of all keys across rows (first-seen order),
    so rows with extra keys (e.g. ``error``) do not break the writer.

    Args:
        rows: Result/metadata records.
        path: Output ``.csv`` path.

    Returns:
        Resolved output path.
    """
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
