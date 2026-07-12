#!/usr/bin/env python3
"""Generate multi-scale question-conditioned candidate pools (Phase 1)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset_loader import iter_manifest
from src.preprocessing.candidate_pool import compute_pool_coverage_stats, generate_candidate_pool
from src.utils.image_io import load_image
from src.utils.logging_utils import log_progress, log_section, setup_experiment_logging
from src.utils.ocr_cache import load_cached_ocr_boxes
from src.utils.paths import outputs_path


def _image_id(record: dict) -> str:
    return record.get("image_id") or record.get("doc_id") or record.get("question_id", "unknown")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate QE-BOPS candidate pools.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logger = setup_experiment_logging("generate_candidates")
    log_section(logger, f"Candidate generation | manifest={args.manifest}")

    rows_out = []
    records = list(iter_manifest(args.manifest))
    if args.limit:
        records = records[: args.limit]

    cand_dir = outputs_path("candidates")
    cand_dir.mkdir(parents=True, exist_ok=True)

    for i, rec in enumerate(records):
        iid = _image_id(rec)
        q = rec.get("question", "")
        log_progress(logger, i + 1, len(records), iid)
        t0 = time.perf_counter()
        image = load_image(rec["image_path"])
        boxes = load_cached_ocr_boxes(iid) or []
        pool = generate_candidate_pool(image, q, boxes)
        stats = compute_pool_coverage_stats(image, pool, boxes)
        edge_hit = 1.0 if any(p.y <= 5 or p.x <= 5 for p in pool) else 0.0
        stats["edge_patch_hit"] = edge_hit
        stats["image_id"] = iid
        stats["question_id"] = rec.get("question_id", "")
        stats["gen_s"] = time.perf_counter() - t0
        rows_out.append(stats)

        out_json = cand_dir / f"{iid}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"patches": [p.as_dict() for p in pool], "question": q}, f)

    stats_csv = outputs_path("candidates", "candidate_pool_stats.csv")
    if rows_out:
        with open(stats_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
    logger.info("Wrote %s (%d rows)", stats_csv, len(rows_out))


if __name__ == "__main__":
    main()
