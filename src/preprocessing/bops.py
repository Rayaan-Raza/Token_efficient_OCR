"""BOPS: Budget-Aware OCR-Guided Overview-Plus-Patch Selection.

Implements the proposed method (Contribution 2): one low-resolution overview
for global layout plus K high-resolution patches selected by OCR-guided scoring.

Supported selection modes:
    - ``ocr_guided``: score candidates with OCR boxes + NMS (default)
    - ``random``: uniform random baseline
    - ``uniform``: evenly spaced grid baseline
    - ``overview_only``: no patches (ablation)

Returns overview image, patch crops, coordinates, and metadata including
patch-budget compliance via :mod:`src.utils.budget_check`.
"""

from __future__ import annotations

import random
from typing import Any

from PIL import Image

from src.ocr.run_ocr import run_ocr_with_boxes
from src.preprocessing.overview import generate_overview
from src.preprocessing.patch_grid import Patch, crop_patch, generate_grid_patches
from src.preprocessing.patch_nms import nms_patches
from src.preprocessing.patch_scoring import score_patch
from src.utils.budget_check import check_patch_budget, merge_budget_fields


def select_random_patches(candidates: list[Patch], k: int, seed: int = 0) -> list[Patch]:
    """Select ``k`` patches uniformly at random (baseline).

    Args:
        candidates: Full candidate grid.
        k: Number of patches to select.
        seed: RNG seed for reproducibility.

    Returns:
        Up to ``k`` patches.
    """
    rng = random.Random(seed)
    return rng.sample(candidates, min(k, len(candidates)))


def select_uniform_patches(candidates: list[Patch], k: int) -> list[Patch]:
    """Select ``k`` evenly spaced patches from the grid (uniform tiling baseline).

    Args:
        candidates: Full candidate grid in row-major order.
        k: Number of patches to select.

    Returns:
        Up to ``k`` patches at regular indices.
    """
    if k >= len(candidates):
        return candidates
    step = len(candidates) / k
    return [candidates[int(i * step)] for i in range(k)]


def select_ocr_guided_patches(
    image: Image.Image,
    candidates: list[Patch],
    k: int,
    ocr_boxes: list[dict[str, Any]] | None = None,
) -> tuple[list[Patch], list[float]]:
    """Score candidates with OCR guidance and apply NMS to pick top ``k``.

    Args:
        image: Full-resolution source image.
        candidates: Patch grid from :func:`generate_grid_patches`.
        k: Patch budget.
        ocr_boxes: Precomputed OCR boxes; if ``None``, runs OCR on a temp file.

    Returns:
        Tuple of (selected patches, their scores).
    """
    if ocr_boxes is None:
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image.save(tmp.name)
        ocr_boxes = run_ocr_with_boxes(tmp.name)
    scores = [score_patch(image, p, ocr_boxes) for p in candidates]
    selected = nms_patches(candidates, scores, iou_threshold=0.5, top_k=k)
    sel_scores = [scores[candidates.index(p)] for p in selected]
    return selected, sel_scores


def run_bops(
    image: Image.Image,
    num_patches: int,
    overview_target_pixels: int = 50_000,
    patch_size: int = 256,
    stride: int = 128,
    mode: str = "ocr_guided",
    seed: int = 0,
) -> dict[str, Any]:
    """Run the full BOPS preprocessing pipeline on one image.

    Args:
        image: Source document/scene image.
        num_patches: Target number of high-res patches (exact budget checked).
        overview_target_pixels: Pixel budget for the low-res overview.
        patch_size: Side length of square patches.
        stride: Grid stride for candidate generation.
        mode: ``ocr_guided``, ``random``, ``uniform``, or ``overview_only``.
        seed: Random seed for ``random`` mode.

    Returns:
        Dict with keys ``overview``, ``patches`` (PIL images), ``patch_coords``,
        and ``meta`` (JSON-serializable metadata including budget fields).
    """
    overview, overview_meta = generate_overview(image, overview_target_pixels)
    candidates = generate_grid_patches(image, patch_size, stride)

    if mode == "random":
        patches = select_random_patches(candidates, num_patches, seed=seed)
    elif mode == "uniform":
        patches = select_uniform_patches(candidates, num_patches)
    elif mode == "overview_only":
        patches = []
    else:
        patches, _ = select_ocr_guided_patches(image, candidates, num_patches)

    patch_images = [crop_patch(image, p) for p in patches]
    meta: dict[str, Any] = {
        "mode": mode,
        "num_patches_target": num_patches,
        "num_patches_actual": len(patches),
        "patches": [p.as_dict() for p in patches],
        **overview_meta,
    }
    budget = check_patch_budget(len(patches), num_patches)
    merge_budget_fields(meta, budget)
    return {
        "overview": overview,
        "patches": patch_images,
        "patch_coords": patches,
        "meta": meta,
    }
