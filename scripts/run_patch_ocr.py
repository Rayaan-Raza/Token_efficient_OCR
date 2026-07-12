#!/usr/bin/env python3
"""OCR all candidate patches and cache text per image."""

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
from src.utils.image_io import load_image
from src.utils.logging_utils import log_progress, log_section, setup_experiment_logging
from src.utils.paths import outputs_path


def _image_id(record: dict) -> str:
    return record.get("image_id") or record.get("doc_id") or record.get("question_id", "unknown")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run patch OCR on candidate pools.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    logger = setup_experiment_logging("patch_ocr")
    log_section(logger, "Patch OCR cache")

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
        log_progress(logger, len(seen), len({ _image_id(r) for r in records }), iid)

        cand_path = cand_dir / f"{iid}.json"
        if not cand_path.exists():
            logger.warning("Missing candidates for %s", iid)
            continue

        with open(cand_path, encoding="utf-8") as f:
            payload = json.load(f)
        patches = [patch_from_dict(p) for p in payload["patches"]]

        image = load_image(rec["image_path"])
        t0 = time.perf_counter()
        patch_ocrs = []
        import tempfile
        for j, p in enumerate(patches):
            crop = crop_patch(image, p)
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            crop.save(tmp.name)
            text = run_ocr_on_image(tmp.name)
            patch_ocrs.append({"index": j, "text": text, **p.as_dict()})

        out = {"image_id": iid, "patches": patch_ocrs}
        with open(out_dir / f"{iid}.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        logger.info("  %d patches OCR'd in %.2fs", len(patch_ocrs), time.perf_counter() - t0)


if __name__ == "__main__":
    main()
