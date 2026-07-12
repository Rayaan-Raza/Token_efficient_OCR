"""Multi-scale question-conditioned candidate patch pool for QE-BOPS."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from PIL import Image

from src.ocr.normalize_text import normalize_text, tokenize
from src.preprocessing.patch_grid import Patch
from src.preprocessing.patch_nms import nms_patches


@dataclass
class CandidatePoolConfig:
    """Configuration for deterministic candidate generation."""

    square_256_stride: int = 128
    square_384_stride: int = 192
    strip_sizes: tuple[tuple[int, int], ...] = ((256, 512), (512, 256))
    strip_overlap: float = 0.5
    edge_anchoring: bool = True
    nms_iou: float = 0.30
    include_ocr_line_centered: bool = True
    include_question_token_centered: bool = True
    include_table_row_strips: bool = True
    include_question_token_centered_in_no_question_ablation: bool = False


def _question_content_tokens(question: str) -> set[str]:
    stop = {
        "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
        "for", "from", "had", "has", "have", "how", "in", "is", "it", "its",
        "of", "on", "or", "that", "the", "this", "to", "was", "were", "what",
        "when", "where", "which", "who", "why", "with",
    }
    return {t for t in tokenize(question) if t not in stop and len(t) > 1}


def _box_center(box: list[list[int]]) -> tuple[float, float]:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _box_area(box: list[list[int]]) -> float:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return max(0.0, (max(xs) - min(xs)) * (max(ys) - min(ys)))


def _centered_patch(cx: float, cy: float, w: int, h: int, img_w: int, img_h: int, idx: int) -> Patch:
    x = int(round(cx - w / 2))
    y = int(round(cy - h / 2))
    return Patch(x, y, w, h, idx).clamp(img_w, img_h)


def _sliding_squares(image: Image.Image, size: int, stride: int, start_idx: int) -> list[Patch]:
    patches: list[Patch] = []
    idx = start_idx
    for y in range(0, image.height, stride):
        for x in range(0, image.width, stride):
            p = Patch(x, y, size, size, idx).clamp(image.width, image.height)
            if p.w > 0 and p.h > 0:
                patches.append(p)
                idx += 1
    return patches


def _sliding_strips(
    image: Image.Image,
    w: int,
    h: int,
    overlap: float,
    start_idx: int,
) -> list[Patch]:
    patches: list[Patch] = []
    idx = start_idx
    step_y = max(1, int(h * (1 - overlap)))
    step_x = max(1, int(w * (1 - overlap)))
    for y in range(0, image.height, step_y):
        for x in range(0, image.width, step_x):
            p = Patch(x, y, w, h, idx).clamp(image.width, image.height)
            if p.w > 0 and p.h > 0:
                patches.append(p)
                idx += 1
    return patches


def _edge_anchored_patches(image: Image.Image, size: int, start_idx: int) -> list[Patch]:
    w, h = image.width, image.height
    anchors = [
        (0, 0),
        (max(0, w - size), 0),
        (0, max(0, h - size)),
        (max(0, w - size), max(0, h - size)),
        (max(0, (w - size) // 2), 0),
        (max(0, (w - size) // 2), max(0, h - size)),
    ]
    out: list[Patch] = []
    idx = start_idx
    for x, y in anchors:
        p = Patch(x, y, size, size, idx).clamp(w, h)
        out.append(p)
        idx += 1
    return out


def _ocr_line_centered(
    image: Image.Image,
    ocr_boxes: list[dict[str, Any]],
    patch_size: int,
    start_idx: int,
) -> list[Patch]:
    patches: list[Patch] = []
    idx = start_idx
    for b in ocr_boxes:
        cx, cy = _box_center(b["box"])
        p = _centered_patch(cx, cy, patch_size, patch_size, image.width, image.height, idx)
        patches.append(p)
        idx += 1
    return patches


def _question_token_centered(
    image: Image.Image,
    ocr_boxes: list[dict[str, Any]],
    question: str,
    patch_size: int,
    start_idx: int,
) -> list[Patch]:
    qtoks = _question_content_tokens(question)
    if not qtoks:
        return []
    patches: list[Patch] = []
    idx = start_idx
    for b in ocr_boxes:
        text = normalize_text(b.get("text", ""))
        btoks = set(tokenize(text))
        if qtoks & btoks:
            cx, cy = _box_center(b["box"])
            p = _centered_patch(cx, cy, patch_size, patch_size, image.width, image.height, idx)
            patches.append(p)
            idx += 1
    return patches


def _table_row_strips(
    image: Image.Image,
    ocr_boxes: list[dict[str, Any]],
    start_idx: int,
) -> list[Patch]:
    """Horizontal strips aligned to OCR row clusters (y-band grouping)."""
    if not ocr_boxes:
        return []
    rows: dict[int, list[dict[str, Any]]] = {}
    for b in ocr_boxes:
        cy = int(_box_center(b["box"])[1])
        band = cy // 20
        rows.setdefault(band, []).append(b)
    patches: list[Patch] = []
    idx = start_idx
    for boxes in rows.values():
        if len(boxes) < 3:
            continue
        xs = [p[0] for b in boxes for p in b["box"]]
        ys = [p[1] for b in boxes for p in b["box"]]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        pad = 8
        w = max(64, x1 - x0 + 2 * pad)
        h = max(32, min(256, y1 - y0 + 2 * pad))
        p = Patch(max(0, x0 - pad), max(0, y0 - pad), w, h, idx).clamp(image.width, image.height)
        patches.append(p)
        idx += 1
    return patches


def generate_original_grid(image: Image.Image, patch_size: int = 256, stride: int = 128) -> list[Patch]:
    """Pilot BOPS 256/128 grid only (bops_original baseline)."""
    return _sliding_squares(image, patch_size, stride, 0)


def generate_candidate_pool(
    image: Image.Image,
    question: str,
    ocr_boxes: list[dict[str, Any]],
    config: CandidatePoolConfig | None = None,
    *,
    include_question_centered: bool = True,
) -> list[Patch]:
    """Build deterministic multi-scale question-conditioned candidate pool."""
    cfg = config or CandidatePoolConfig()
    patches: list[Patch] = []
    idx = 0

    patches.extend(_sliding_squares(image, 256, cfg.square_256_stride, idx))
    idx = len(patches)
    patches.extend(_sliding_squares(image, 384, cfg.square_384_stride, idx))
    idx = len(patches)

    for sw, sh in cfg.strip_sizes:
        patches.extend(_sliding_strips(image, sw, sh, cfg.strip_overlap, idx))
        idx = len(patches)

    if cfg.edge_anchoring:
        patches.extend(_edge_anchored_patches(image, 256, idx))
        idx = len(patches)

    if cfg.include_ocr_line_centered and ocr_boxes:
        patches.extend(_ocr_line_centered(image, ocr_boxes, 256, idx))
        idx = len(patches)

    if include_question_centered and cfg.include_question_token_centered and question and ocr_boxes:
        patches.extend(_question_token_centered(image, ocr_boxes, question, 256, idx))
        idx = len(patches)

    if cfg.include_table_row_strips and ocr_boxes:
        patches.extend(_table_row_strips(image, ocr_boxes, idx))
        idx = len(patches)

    if not patches:
        return patches

    scores = [1.0] * len(patches)
    kept = nms_patches(patches, scores, iou_threshold=cfg.nms_iou, top_k=len(patches))
    for i, p in enumerate(kept):
        p.index = i
    return kept


def pool_hash(patches: list[Patch], question_id: str = "") -> str:
    """Stable hash for fairness assertions."""
    payload = {
        "question_id": question_id,
        "patches": [p.as_dict() for p in patches],
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def compute_pool_coverage_stats(
    image: Image.Image,
    patches: list[Patch],
    ocr_boxes: list[dict[str, Any]],
    *,
    coverage_sample_step: int = 8,
) -> dict[str, float]:
    """G1 metrics: union area coverage, candidate area ratio, OCR-box coverage."""
    img_area = max(1, image.width * image.height)
    step = coverage_sample_step
    sample_xs = list(range(0, image.width, step))
    sample_ys = list(range(0, image.height, step))
    sample_total = max(1, len(sample_xs) * len(sample_ys))

    covered = [[False] * image.width for _ in range(image.height)]
    for p in patches:
        for y in range(p.y, min(image.height, p.y + p.h), step):
            for x in range(p.x, min(image.width, p.x + p.w), step):
                covered[y][x] = True
    sample_hits = sum(covered[y][x] for y in sample_ys for x in sample_xs)
    unique_image_area_coverage = min(1.0, sample_hits / sample_total)
    candidate_area_ratio = sum(max(0, p.w) * max(0, p.h) for p in patches) / img_area

    def _center_in_any(box: list[list[int]]) -> bool:
        cx, cy = _box_center(box)
        ix, iy = int(cx), int(cy)
        for p in patches:
            if p.x <= ix < p.x + p.w and p.y <= iy < p.y + p.h:
                return True
        return False

    if not ocr_boxes:
        return {
            "unique_image_area_coverage": float(unique_image_area_coverage),
            "candidate_area_ratio": float(candidate_area_ratio),
            "ocr_box_center_coverage": 0.0,
            "small_box_coverage": 0.0,
            "num_candidates": float(len(patches)),
        }

    areas = [_box_area(b["box"]) for b in ocr_boxes]
    p25 = sorted(areas)[max(0, len(areas) // 4)] if areas else 0.0
    centers = sum(1 for b in ocr_boxes if _center_in_any(b["box"]))
    small = [b for b, a in zip(ocr_boxes, areas) if a <= p25 or a <= 500]
    small_hit = sum(1 for b in small if _center_in_any(b["box"]))

    return {
        "unique_image_area_coverage": float(unique_image_area_coverage),
        "candidate_area_ratio": float(candidate_area_ratio),
        "ocr_box_center_coverage": centers / len(ocr_boxes),
        "small_box_coverage": (small_hit / len(small)) if small else 1.0,
        "num_candidates": float(len(patches)),
    }
