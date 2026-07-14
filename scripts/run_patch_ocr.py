#!/usr/bin/env python3
"""OCR all candidate patches and cache text per image.

Modes:
  default: EasyOCR each crop (slow, high quality)
  --from-fullpage-boxes: derive patch text from cached full-page OCR boxes
                         intersecting the patch (fast; used to scale to n=500)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset_loader import iter_manifest
from src.metrics.answer_coverage import patch_from_dict
from src.ocr.run_ocr import run_ocr_on_image
from src.preprocessing.patch_grid import crop_patch
from src.preprocessing.patch_scoring_qa import patch_ocr_text
from src.utils.image_io import load_image
from src.utils.logging_utils import log_progress, log_section, setup_experiment_logging
from src.utils.ocr_cache import load_cached_ocr_boxes
from src.utils.paths import outputs_path


def _image_id(record: dict) -> str:
    return record.get("image_id") or record.get("doc_id") or record.get("question_id", "unknown")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run patch OCR on candidate pools.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--from-fullpage-boxes",
        action="store_true",
        help="Fill patch text from full-page OCR box intersection (no per-crop OCR).",
    )
    args = parser.parse_args()

    logger = setup_experiment_logging("patch_ocr")
    mode = "fullpage-boxes" if args.from_fullpage_boxes else "easyocr-crops"
    log_section(logger, f"Patch OCR cache | mode={mode}")

    records = list(iter_manifest(args.manifest))
    if args.limit:
        records = records[: args.limit]

    out_dir = outputs_path("ocr", "patches")
    out_dir.mkdir(parents=True, exist_ok=True)
    cand_dir = outputs_path("candidates")

    seen = set()
    for rec in records:
        iid = _image_id(rec)
        if iid in seen:
            continue
        seen.add(iid)
        log_progress(logger, len(seen), len({_image_id(r) for r in records}), iid)

        cand_path = cand_dir / f"{iid}.json"
        if not cand_path.exists():
            logger.warning("Missing candidates for %s", iid)
            continue

        out_path = out_dir / f"{iid}.json"
        if out_path.exists():
            logger.info("  cache hit")
            continue

        with open(cand_path, encoding="utf-8") as f:
            payload = json.load(f)
        patches = [patch_from_dict(p) for p in payload["patches"]]

        t0 = time.perf_counter()
        patch_ocrs = []
        if args.from_fullpage_boxes:
            boxes = load_cached_ocr_boxes(iid) or []
            for j, p in enumerate(patches):
                text = patch_ocr_text(p, boxes)
                idx = int(getattr(p, "index", j))
                patch_ocrs.append({"index": idx, "text": text, **p.as_dict(), "text_source": "fullpage_boxes"})
        else:
            image = load_image(rec["image_path"])
            import tempfile
            for j, p in enumerate(patches):
                crop = crop_patch(image, p)
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                crop.save(tmp.name)
                text = run_ocr_on_image(tmp.name)
                idx = int(getattr(p, "index", j))
                patch_ocrs.append({"index": idx, "text": text, **p.as_dict(), "text_source": "easyocr"})

        out = {"image_id": iid, "patches": patch_ocrs, "mode": mode}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        logger.info("  %d patches in %.2fs", len(patch_ocrs), time.perf_counter() - t0)


if __name__ == "__main__":
    main()
