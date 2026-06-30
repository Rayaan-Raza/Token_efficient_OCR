"""Non-maximum suppression for overlapping patches."""

from __future__ import annotations

from src.preprocessing.patch_grid import Patch


def iou(a: Patch, b: Patch) -> float:
    x0 = max(a.x, b.x)
    y0 = max(a.y, b.y)
    x1 = min(a.x + a.w, b.x + b.w)
    y1 = min(a.y + a.h, b.y + b.h)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def nms_patches(
    patches: list[Patch],
    scores: list[float],
    iou_threshold: float = 0.5,
    top_k: int | None = None,
) -> list[Patch]:
    if not patches:
        return []
    order = sorted(range(len(patches)), key=lambda i: scores[i], reverse=True)
    kept: list[Patch] = []
    kept_scores: list[float] = []
    while order:
        i = order.pop(0)
        kept.append(patches[i])
        kept_scores.append(scores[i])
        if top_k and len(kept) >= top_k:
            break
        order = [j for j in order if iou(patches[i], patches[j]) < iou_threshold]
    return kept
