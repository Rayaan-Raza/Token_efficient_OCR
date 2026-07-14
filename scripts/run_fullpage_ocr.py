#!/usr/bin/env python3
"""Cache full-page OCR boxes for DocVQA samples."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset_loader import iter_manifest
from src.ocr.run_ocr import reset_ocr, run_ocr_with_boxes
from src.utils.image_io import load_image
from src.utils.logging_utils import log_progress, log_section, setup_experiment_logging
from src.utils.ocr_cache import load_cached_ocr_boxes, save_cached_ocr_boxes
from src.utils.paths import outputs_path


def _image_id(record: dict) -> str:
    return record.get("image_id") or record.get("doc_id") or record.get("question_id", "unknown")


def _ocr_image(image, max_side: int = 1600) -> list[dict]:
    """OCR with optional downscale to reduce GPU memory; OOM falls back to CPU."""
    import tempfile

    work = image
    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        work = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    work.save(tmp.name)
    try:
        return run_ocr_with_boxes(tmp.name)
    except Exception as exc:
        msg = str(exc).lower()
        if "out of memory" in msg or "cuda" in msg:
            reset_ocr(force_cpu=True)
            return run_ocr_with_boxes(tmp.name)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full-page OCR cache.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--engine", default="easyocr")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-side", type=int, default=1600, help="Downscale long side before OCR")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logger = setup_experiment_logging("fullpage_ocr")
    log_section(logger, f"Full-page OCR | engine={args.engine} max_side={args.max_side}")

    if args.force_cpu:
        reset_ocr(force_cpu=True)

    records = list(iter_manifest(args.manifest))
    if args.limit:
        records = records[: args.limit]

    seen = set()
    for i, rec in enumerate(records):
        iid = _image_id(rec)
        if iid in seen:
            continue
        seen.add(iid)
        log_progress(logger, len(seen), len({_image_id(r) for r in records}), iid)
        if load_cached_ocr_boxes(iid):
            logger.info("  cache hit")
            continue
        t0 = time.perf_counter()
        image = load_image(rec["image_path"])
        boxes = _ocr_image(image, max_side=args.max_side)
        save_cached_ocr_boxes(iid, boxes)
        logger.info("  OCR %.2fs | %d boxes", time.perf_counter() - t0, len(boxes))

    logger.info("Cache dir: %s", outputs_path("cache", "ocr_boxes"))


if __name__ == "__main__":
    main()
