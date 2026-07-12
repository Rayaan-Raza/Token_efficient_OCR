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
from src.ocr.run_ocr import run_ocr_with_boxes
from src.utils.image_io import load_image
from src.utils.logging_utils import log_progress, log_section, setup_experiment_logging
from src.utils.ocr_cache import load_cached_ocr_boxes, save_cached_ocr_boxes
from src.utils.paths import outputs_path


def _image_id(record: dict) -> str:
    return record.get("image_id") or record.get("doc_id") or record.get("question_id", "unknown")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full-page OCR cache.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--engine", default="easyocr")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logger = setup_experiment_logging("fullpage_ocr")
    log_section(logger, f"Full-page OCR | engine={args.engine}")

    records = list(iter_manifest(args.manifest))
    if args.limit:
        records = records[: args.limit]

    seen = set()
    for i, rec in enumerate(records):
        iid = _image_id(rec)
        if iid in seen:
            continue
        seen.add(iid)
        log_progress(logger, len(seen), len({ _image_id(r) for r in records }), iid)
        if load_cached_ocr_boxes(iid):
            logger.info("  cache hit")
            continue
        t0 = time.perf_counter()
        image = load_image(rec["image_path"])
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image.save(tmp.name)
        boxes = run_ocr_with_boxes(tmp.name)
        save_cached_ocr_boxes(iid, boxes)
        logger.info("  OCR %.2fs | %d boxes", time.perf_counter() - t0, len(boxes))

    logger.info("Cache dir: %s", outputs_path("cache", "ocr_boxes"))


if __name__ == "__main__":
    main()
