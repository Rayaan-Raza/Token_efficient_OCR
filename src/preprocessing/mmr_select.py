"""MMR-style diverse patch selection for QE-BOPS."""

from __future__ import annotations

from src.preprocessing.patch_grid import Patch


def _iou(a: Patch, b: Patch) -> float:
    ix0 = max(a.x, b.x)
    iy0 = max(a.y, b.y)
    ix1 = min(a.x + a.w, b.x + b.w)
    iy1 = min(a.y + a.h, b.y + b.h)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = a.w * a.h + b.w * b.h - inter
    return inter / max(1, union)


def patch_iou(a: Patch, b: Patch) -> float:
    return _iou(a, b)


def mmr_select(
    patches: list[Patch],
    scores: list[float],
    k: int,
    *,
    lambda_: float = 0.75,
) -> list[Patch]:
    """Select K patches with MMR diversity."""
    if k >= len(patches):
        return list(patches)
    if not patches:
        return []

    max_score = max(scores) if scores else 1.0
    norm = [s / max_score if max_score > 0 else 0.0 for s in scores]
    selected: list[Patch] = []
    selected_idx: list[int] = []
    remaining = set(range(len(patches)))

    while len(selected) < k and remaining:
        best_i = None
        best_mmr = float("-inf")
        for i in remaining:
            relevance = norm[i]
            if selected_idx:
                max_iou = max(_iou(patches[i], patches[j]) for j in selected_idx)
            else:
                max_iou = 0.0
            mmr = lambda_ * relevance - (1 - lambda_) * max_iou
            if mmr > best_mmr:
                best_mmr = mmr
                best_i = i
        if best_i is None:
            break
        selected.append(patches[best_i])
        selected_idx.append(best_i)
        remaining.remove(best_i)
    return selected
