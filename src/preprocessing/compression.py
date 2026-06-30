"""JPEG/WebP compression baselines under byte budgets.

Binary-searches encoder quality so the output file size is at or below the
target byte budget (±2% enforced in metadata). Used as strong baselines in
OCR and VLM experiments alongside resize and BOPS.
"""

from __future__ import annotations

import io
from typing import Literal

from PIL import Image

from src.utils.budget_check import check_byte_budget, merge_budget_fields

Format = Literal["JPEG", "WEBP"]


def compress_to_byte_budget(
    image: Image.Image,
    target_bytes: int,
    fmt: Format = "JPEG",
    min_quality: int = 5,
    max_quality: int = 95,
) -> tuple[bytes, dict]:
    """Encode an image to bytes, targeting a maximum file size.

  Uses binary search over quality settings. If no quality fits under the budget,
  returns the lowest-quality encoding.

    Args:
        image: Source PIL image.
        target_bytes: Maximum allowed encoded size in bytes.
        fmt: ``"JPEG"`` or ``"WEBP"``.
        min_quality: Lower bound for quality search.
        max_quality: Upper bound for quality search.

    Returns:
        Tuple of (encoded bytes, metadata including quality and budget fields).
    """
    low, high = min_quality, max_quality
    best_data = b""
    best_q = min_quality
    while low <= high:
        mid = (low + high) // 2
        buf = io.BytesIO()
        save_kwargs = {"format": fmt, "quality": mid}
        if fmt == "JPEG":
            save_kwargs["optimize"] = True
        image.save(buf, **save_kwargs)
        data = buf.getvalue()
        if len(data) <= target_bytes:
            best_data = data
            best_q = mid
            low = mid + 1
        else:
            high = mid - 1
    if not best_data:
        buf = io.BytesIO()
        image.save(buf, format=fmt, quality=min_quality)
        best_data = buf.getvalue()
        best_q = min_quality
    meta = {
        "method": fmt.lower(),
        "quality": best_q,
        "target_bytes": target_bytes,
        "actual_bytes": len(best_data),
    }
    budget = check_byte_budget(len(best_data), target_bytes)
    merge_budget_fields(meta, budget)
    return best_data, meta


def compress_image_to_file(
    image: Image.Image,
    out_path: str,
    target_bytes: int,
    fmt: Format = "JPEG",
) -> dict:
    """Compress an image and write it to disk.

    Args:
        image: Source PIL image.
        out_path: Destination file path.
        target_bytes: Byte budget for the encoded file.
        fmt: ``"JPEG"`` or ``"WEBP"``.

    Returns:
        Metadata dict from :func:`compress_to_byte_budget` plus ``output_path``.
    """
    data, meta = compress_to_byte_budget(image, target_bytes, fmt=fmt)
    with open(out_path, "wb") as f:
        f.write(data)
    meta["output_path"] = out_path
    return meta
