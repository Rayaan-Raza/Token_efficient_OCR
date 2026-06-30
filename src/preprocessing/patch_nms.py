"""Non-maximum suppression (NMS) for overlapping patch candidates.

After scoring, many high-scoring patches may overlap heavily. NMS keeps the
highest-scoring patch and suppresses others with IoU above a threshold, then
repeats until ``top_k`` patches are selected or candidates are exhausted.
"""

from __future__ import annotations

from src.preprocessing.patch_grid import Patch


def iou(a: Patch, b: Patch) -> float:
    """Intersection-over-union for two axis-aligned patches.

    Args:
        a: First patch.
        b: Second patch.

    Returns:
        IoU in [0, 1].
    """
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
    """Greedy NMS: keep highest-scoring non-overlapping patches.

    Args:
        patches: Candidate patches (same order as ``scores``).
        scores: Importance score per patch.
        iou_threshold: Suppress patches with IoU >= this value to a kept patch.
        top_k: Stop after selecting this many patches (``None`` = no limit).

    Returns:
        Subset of ``patches`` in descending score order.
    """
    if not patches:
        return []
    order = sorted(range(len(patches)), key=lambda i: scores[i], reverse=True)
    kept: list[Patch] = []
    while order:
        i = order.pop(0)
        kept.append(patches[i])
        if top_k and len(kept) >= top_k:
            break
        order = [j for j in order if iou(patches[i], patches[j]) < iou_threshold]
    return kept
