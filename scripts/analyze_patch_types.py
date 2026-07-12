#!/usr/bin/env python3
"""Classify selected patch types by method."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset_loader import iter_manifest
from src.preprocessing.selectors import select_patches
from src.utils.image_io import load_image
from src.utils.logging_utils import setup_experiment_logging
from src.utils.ocr_cache import load_cached_ocr_boxes
from src.utils.paths import outputs_path


def _classify_patch(y: int, h: int, img_h: int, ocr_text: str) -> str:
    if not ocr_text.strip():
        return "blank"
    if y < img_h * 0.15:
        return "header"
    if y > img_h * 0.85:
        return "footer"
    if ":" in ocr_text and len(ocr_text) < 40:
        return "label_value"
    if len(ocr_text) < 15:
        return "form_field"
    return "paragraph"


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch type distribution.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--methods", default="bops_fair_pool,qe_bops")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    logger = setup_experiment_logging("patch_types")
    records = list(iter_manifest(args.manifest))[: args.limit]
    rows = []
    for method in args.methods.split(","):
        counts: dict[str, int] = {}
        for rec in records:
            img = load_image(rec["image_path"])
            boxes = load_cached_ocr_boxes(rec["image_id"]) or []
            sel = select_patches(img, method.strip(), args.k, rec.get("question", ""), boxes)
            for p in sel.patches:
                from src.preprocessing.patch_scoring_qa import patch_ocr_text
                text = patch_ocr_text(p, boxes)
                t = _classify_patch(p.y, p.h, img.height, text)
                counts[t] = counts.get(t, 0) + 1
        for t, c in counts.items():
            rows.append({"method": method.strip(), "patch_type": t, "count": c})

    out = outputs_path("metrics", "patch_type_distribution.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "patch_type", "count"])
        w.writeheader()
        w.writerows(rows)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
