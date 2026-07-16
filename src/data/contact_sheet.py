"""Compose page images into a contact sheet for transfer experiments.

This helper is used by the MP-DocVQA contact-sheet transfer setting (not the
standard MP-DocVQA benchmark) to build a single image per document under a
max-side budget.
"""

from __future__ import annotations

import math
from typing import Iterable

from PIL import Image


def build_contact_sheet(
    pages: Iterable[Image.Image],
    *,
    max_side: int = 2048,
    padding: int = 10,
    background: tuple[int, int, int] = (255, 255, 255),
) -> tuple[Image.Image, dict]:
    """Arrange pages into a grid contact sheet with a max-side budget.

    Args:
        pages: Iterable of PIL images (one per page).
        max_side: Maximum width/height for the output contact sheet.
        padding: Pixel padding between pages and at borders.
        background: RGB fill color for the sheet.

    Returns:
        Tuple of (contact sheet image, metadata dict).
    """
    page_list = [page.convert("RGB") for page in pages]
    if not page_list:
        raise ValueError("build_contact_sheet requires at least one page image.")

    count = len(page_list)
    cols = max(1, math.ceil(math.sqrt(count)))
    rows = max(1, math.ceil(count / cols))

    max_side = int(max_side)
    padding = int(padding)
    cell_w = max(1, (max_side - padding * (cols + 1)) // cols)
    cell_h = max(1, (max_side - padding * (rows + 1)) // rows)

    sheet_w = padding * (cols + 1) + cell_w * cols
    sheet_h = padding * (rows + 1) + cell_h * rows
    sheet = Image.new("RGB", (sheet_w, sheet_h), background)

    for idx, page in enumerate(page_list):
        scale = min(cell_w / page.width, cell_h / page.height, 1.0)
        new_w = max(1, int(page.width * scale))
        new_h = max(1, int(page.height * scale))
        if new_w != page.width or new_h != page.height:
            page = page.resize((new_w, new_h), Image.Resampling.LANCZOS)
        row = idx // cols
        col = idx % cols
        x0 = padding + col * (cell_w + padding) + (cell_w - new_w) // 2
        y0 = padding + row * (cell_h + padding) + (cell_h - new_h) // 2
        sheet.paste(page, (x0, y0))

    metadata = {
        "page_count": count,
        "grid_rows": rows,
        "grid_cols": cols,
        "cell_size": [cell_w, cell_h],
        "sheet_size": [sheet_w, sheet_h],
        "max_side": max_side,
        "max_side_exceeded": max(sheet_w, sheet_h) > max_side,
        "method": "contact_sheet",
    }
    return sheet, metadata
