#!/usr/bin/env python3
"""Run VLM DocVQA evaluation."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset_loader import iter_manifest
from src.preprocessing.bops import run_bops
from src.preprocessing.resize import resize_to_area_ratio
from src.utils.image_io import load_image, write_metadata_csv
from src.utils.paths import outputs_path, repo_path
from src.vlm.qa_metrics import anls, exact_match
from src.vlm.run_vlm import run_vlm_overview_patches, run_vlm_single


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--method", default="bops", choices=["resize", "overview_only", "random", "uniform", "bops"])
    parser.add_argument("--num-patches", type=int, default=4)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="Skip model load; write placeholder answers")
    args = parser.parse_args()

    rows = []
    for i, record in enumerate(iter_manifest(args.manifest)):
        if i >= args.limit:
            break
        image = load_image(repo_path(record["image_path"]))
        question = record.get("question", "")
        answers = record.get("answer", [])
        if isinstance(answers, str):
            answers = [answers]
        t0 = time.perf_counter()
        if args.dry_run:
            pred = "dry-run"
        elif args.method == "resize":
            resized, _ = resize_to_area_ratio(image, 0.25)
            pred = run_vlm_single(resized, question)
        elif args.method == "overview_only":
            result = run_bops(image, 0, mode="overview_only")
            pred = run_vlm_single(result["overview"], question)
        else:
            mode = "ocr_guided" if args.method == "bops" else args.method
            result = run_bops(image, args.num_patches, mode=mode)
            pred = run_vlm_overview_patches(result["overview"], result["patches"], question)
        rows.append({
            "image_id": record["image_id"],
            "method": args.method,
            "num_patches": args.num_patches,
            "prediction": pred,
            "exact_match": exact_match(pred, answers),
            "anls": anls(pred, answers),
            "runtime_sec": round(time.perf_counter() - t0, 3),
        })

    csv_path = outputs_path("metrics", "vlm_metrics.csv")
    write_metadata_csv(rows, csv_path)
    print(f"Wrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
