"""OCR-protected full-page compression for budgeted Document VQA.

Variants (all finish at the same pixel budget as the resize baseline:
``0.25 * original_area``):

- ``margin_crop``: crop non-white content bbox then budget-resize
- ``ws_compress``: remove mostly blank bands while protecting OCR rows/cols
- ``ocr_seam``: OCR-protected seam carving (no seams through OCR boxes)

Leakage: uses page OCR boxes only; never gold answers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
from PIL import Image

from src.utils.budget_check import check_pixel_budget, merge_budget_fields
from src.utils.image_io import image_area

Variant = Literal["margin_crop", "ws_compress", "ocr_seam"]

METHOD_LABELS = {
    "margin_crop": "margin_crop_resize",
    "ws_compress": "ws_compress_resize",
    "ocr_seam": "ocr_seam_resize",
}


@dataclass
class CompressResult:
    image: Image.Image
    meta: dict[str, Any]


def _to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _box_aabb(box: Sequence[Sequence[float]]) -> tuple[int, int, int, int]:
    xs = [float(p[0]) for p in box]
    ys = [float(p[1]) for p in box]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def _ocr_mask(
    h: int,
    w: int,
    ocr_boxes: Sequence[dict[str, Any]] | None,
    *,
    pad: int = 2,
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    if not ocr_boxes:
        return mask
    for item in ocr_boxes:
        box = item.get("box")
        if not box:
            continue
        x0, y0, x1, y1 = _box_aabb(box)
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(w - 1, x1 + pad)
        y1 = min(h - 1, y1 + pad)
        if x1 >= x0 and y1 >= y0:
            mask[y0 : y1 + 1, x0 : x1 + 1] = True
    return mask


def _remap_boxes_after_crop(
    ocr_boxes: Sequence[dict[str, Any]] | None,
    bbox: list[int] | None,
) -> list[dict[str, Any]] | None:
    if not ocr_boxes or not bbox:
        return list(ocr_boxes) if ocr_boxes else None
    x0, y0, _, _ = bbox
    out: list[dict[str, Any]] = []
    for item in ocr_boxes:
        box = item.get("box")
        if not box:
            continue
        out.append({
            **item,
            "box": [[float(p[0]) - x0, float(p[1]) - y0] for p in box],
        })
    return out


def margin_crop(
    image: Image.Image,
    *,
    white_thresh: int = 245,
    pad: int = 8,
) -> tuple[Image.Image, dict[str, Any]]:
    """Crop near-white margins; fallback to original if empty."""
    arr = _to_rgb_array(image)
    content = np.any(arr < white_thresh, axis=2)
    if not content.any():
        return image.copy(), {"cropped": False, "reason": "no_content"}
    ys, xs = np.where(content)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(arr.shape[0], int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(arr.shape[1], int(xs.max()) + 1 + pad)
    cropped = image.crop((x0, y0, x1, y1))
    return cropped, {
        "cropped": True,
        "bbox": [x0, y0, x1, y1],
        "orig_size": [image.width, image.height],
        "crop_size": [cropped.width, cropped.height],
    }


def whitespace_compress(
    image: Image.Image,
    ocr_boxes: Sequence[dict[str, Any]] | None,
    *,
    white_thresh: int = 245,
    blank_frac: float = 0.98,
    keep_every: int = 4,
) -> tuple[Image.Image, dict[str, Any]]:
    """Remove mostly blank rows/cols while protecting OCR-covered lines."""
    arr = _to_rgb_array(image)
    h, w = arr.shape[:2]
    protect = _ocr_mask(h, w, ocr_boxes, pad=3)
    content = np.any(arr < white_thresh, axis=2)
    row_blank = (content.mean(axis=1) < (1.0 - blank_frac)) & (~protect.any(axis=1))
    col_blank = (content.mean(axis=0) < (1.0 - blank_frac)) & (~protect.any(axis=0))

    keep_rows = np.ones(h, dtype=bool)
    blank_idx = np.where(row_blank)[0]
    if blank_idx.size:
        keep_rows[blank_idx] = False
        keep_rows[blank_idx[::keep_every]] = True
        keep_rows[protect.any(axis=1)] = True

    keep_cols = np.ones(w, dtype=bool)
    blank_c = np.where(col_blank)[0]
    if blank_c.size:
        keep_cols[blank_c] = False
        keep_cols[blank_c[::keep_every]] = True
        keep_cols[protect.any(axis=0)] = True

    if int(keep_rows.sum()) < max(8, h // 10) or int(keep_cols.sum()) < max(8, w // 10):
        return image.copy(), {
            "compressed": False,
            "reason": "protect_floor",
            "rows_removed": 0,
            "cols_removed": 0,
        }

    out = arr[keep_rows][:, keep_cols]
    return Image.fromarray(out), {
        "compressed": True,
        "rows_removed": int((~keep_rows).sum()),
        "cols_removed": int((~keep_cols).sum()),
        "out_size": [int(out.shape[1]), int(out.shape[0])],
    }


def _energy_map(arr: np.ndarray, protect: np.ndarray) -> np.ndarray:
    gray = np.asarray(Image.fromarray(arr).convert("L"), dtype=np.float32)
    gy, gx = np.gradient(gray)
    energy = np.hypot(gx, gy).astype(np.float64)
    energy += (255.0 - gray) * 0.15
    energy[protect] = 1e9
    return energy


def _min_seam_vertical(energy: np.ndarray) -> np.ndarray:
    """Return row→col seam indices for a vertical seam."""
    h, w = energy.shape
    dp = energy.copy()
    back = np.zeros((h, w), dtype=np.int16)
    for i in range(1, h):
        left = np.r_[dp[i - 1, 0], dp[i - 1, :-1]]
        mid = dp[i - 1]
        right = np.r_[dp[i - 1, 1:], dp[i - 1, -1]]
        stacked = np.vstack([left, mid, right])
        choice = np.argmin(stacked, axis=0)
        dp[i] += stacked[choice, np.arange(w)]
        # Map choice 0/1/2 -> j-1 / j / j+1
        idxs = np.arange(w)
        back[i] = np.clip(idxs + (choice - 1), 0, w - 1).astype(np.int16)
    seam = np.zeros(h, dtype=np.int32)
    j = int(np.argmin(dp[-1]))
    for i in range(h - 1, -1, -1):
        seam[i] = j
        j = int(back[i, j]) if i > 0 else j
    return seam


def _remove_vertical_seam(
    arr: np.ndarray, protect: np.ndarray, seam: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    h, w, c = arr.shape
    out = np.zeros((h, w - 1, c), dtype=arr.dtype)
    pout = np.zeros((h, w - 1), dtype=bool)
    for i in range(h):
        j = int(seam[i])
        out[i, :j] = arr[i, :j]
        out[i, j:] = arr[i, j + 1 :]
        pout[i, :j] = protect[i, :j]
        pout[i, j:] = protect[i, j + 1 :]
    return out, pout


def _remove_horizontal_seam(
    arr: np.ndarray, protect: np.ndarray, seam: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    arr_t = np.transpose(arr, (1, 0, 2))
    prot_t = protect.T
    out_t, pout_t = _remove_vertical_seam(arr_t, prot_t, seam)
    return np.transpose(out_t, (1, 0, 2)), pout_t.T


def _maybe_downscale_for_carve(
    image: Image.Image,
    ocr_boxes: Sequence[dict[str, Any]] | None,
    max_side: int = 1280,
) -> tuple[Image.Image, list[dict[str, Any]] | None, float]:
    m = max(image.size)
    if m <= max_side:
        return image, list(ocr_boxes) if ocr_boxes else None, 1.0
    scale = max_side / float(m)
    nw = max(1, int(image.width * scale))
    nh = max(1, int(image.height * scale))
    small = image.resize((nw, nh), Image.Resampling.BILINEAR)
    boxes = None
    if ocr_boxes:
        boxes = []
        for item in ocr_boxes:
            box = item.get("box")
            if not box:
                continue
            boxes.append({
                **item,
                "box": [[float(p[0]) * scale, float(p[1]) * scale] for p in box],
            })
    return small, boxes, scale


def ocr_protected_seam_carve(
    image: Image.Image,
    ocr_boxes: Sequence[dict[str, Any]] | None,
    *,
    target_frac: float = 0.88,
    max_seams: int = 60,
    carve_max_side: int = 1280,
) -> tuple[Image.Image, dict[str, Any]]:
    """Remove low-energy seams avoiding OCR boxes until size ~ target_frac."""
    work, boxes, scale = _maybe_downscale_for_carve(
        image, ocr_boxes, max_side=carve_max_side
    )
    arr = _to_rgb_array(work)
    h, w = arr.shape[:2]
    protect = _ocr_mask(h, w, boxes, pad=2)
    if protect.mean() > 0.85:
        return image.copy(), {
            "carved": False,
            "reason": "ocr_too_dense",
            "v_seams": 0,
            "h_seams": 0,
            "work_scale": scale,
        }

    target_w = max(32, int(w * target_frac))
    target_h = max(32, int(h * target_frac))
    v_removed = 0
    h_removed = 0

    while arr.shape[1] > target_w and v_removed < max_seams:
        energy = _energy_map(arr, protect)
        seam = _min_seam_vertical(energy)
        if any(bool(protect[i, int(seam[i])]) for i in range(arr.shape[0])):
            break
        seam_e = float(np.mean([energy[i, int(seam[i])] for i in range(arr.shape[0])]))
        if seam_e > 1e8:
            break
        arr, protect = _remove_vertical_seam(arr, protect, seam)
        v_removed += 1

    while arr.shape[0] > target_h and h_removed < max_seams:
        energy_t = _energy_map(arr, protect).T
        seam = _min_seam_vertical(energy_t)
        if any(bool(protect[int(seam[j]), j]) for j in range(arr.shape[1])):
            break
        seam_e = float(
            np.mean([energy_t[j, int(seam[j])] for j in range(arr.shape[1])])
        )
        if seam_e > 1e8:
            break
        arr, protect = _remove_horizontal_seam(arr, protect, seam)
        h_removed += 1

    carved = Image.fromarray(arr)
    # If we carved on a downscaled copy, scale back toward original content size
    # before the shared budget resize (preserves relative compression).
    if scale < 1.0 and (v_removed or h_removed):
        back_w = max(1, int(carved.width / scale))
        back_h = max(1, int(carved.height / scale))
        carved = carved.resize((back_w, back_h), Image.Resampling.LANCZOS)

    return carved, {
        "carved": bool(v_removed or h_removed),
        "v_seams": v_removed,
        "h_seams": h_removed,
        "out_size": [carved.width, carved.height],
        "work_scale": scale,
        "orig_size": [image.width, image.height],
    }


def resize_to_original_area_ratio(
    image: Image.Image,
    original: Image.Image,
    area_ratio: float,
) -> tuple[Image.Image, dict[str, Any]]:
    """Resize ``image`` so its area ≈ ``area_ratio * original.area``."""
    orig_area = image_area(original)
    target_area = orig_area * area_ratio
    cur_area = max(1, image_area(image))
    scale = math.sqrt(target_area / cur_area)
    new_w = max(1, int(image.width * scale))
    new_h = max(1, int(image.height * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    meta = {
        "method": "budget_resize",
        "area_ratio": area_ratio,
        "original_area": orig_area,
        "source_area": cur_area,
        "actual_area": image_area(resized),
    }
    budget = check_pixel_budget(image_area(resized), int(target_area))
    merge_budget_fields(meta, budget)
    return resized, meta


def compress_fullpage(
    image: Image.Image,
    variant: Variant,
    *,
    ocr_boxes: Sequence[dict[str, Any]] | None = None,
    area_ratio: float = 0.25,
) -> CompressResult:
    """Apply a compression variant then budget-match the resize baseline."""
    steps: dict[str, Any] = {"variant": variant}
    working, crop_meta = margin_crop(image)
    steps["margin_crop"] = crop_meta
    boxes = _remap_boxes_after_crop(
        ocr_boxes, crop_meta.get("bbox") if crop_meta.get("cropped") else None
    )

    if variant == "margin_crop":
        pass
    elif variant == "ws_compress":
        working, meta = whitespace_compress(working, boxes)
        steps["ws_compress"] = meta
    elif variant == "ocr_seam":
        working, meta = ocr_protected_seam_carve(working, boxes)
        steps["ocr_seam"] = meta
    else:
        raise ValueError(f"unknown variant: {variant}")

    resized, rmeta = resize_to_original_area_ratio(working, image, area_ratio)
    steps["resize"] = rmeta
    steps["method"] = METHOD_LABELS[variant]
    steps["invalid_budget"] = bool(rmeta.get("invalid_budget", False))
    return CompressResult(image=resized, meta=steps)
