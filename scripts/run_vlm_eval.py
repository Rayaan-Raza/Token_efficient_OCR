#!/usr/bin/env python3
"""Run VLM DocVQA evaluation (Phases 9–10).

Evaluates document QA under preprocessing methods. Writes per-method CSV files
(default) so comparisons are not lost across runs. Use ``merge_vlm_metrics.py``
to combine into a single table.

Example::

    python scripts/run_vlm_eval.py \\
        --manifest data/manifests/docvqa_debug.jsonl \\
        --method bops --num-patches 2 --limit 3 \\
        --experiment-stage sanity
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
from src.preprocessing.bops import run_bops
from src.preprocessing.resize import resize_to_area_ratio
from src.utils.experiment_io import (
    default_vlm_metrics_path,
    iso_timestamp,
    new_run_id,
    serialize_answers,
    write_or_append_csv,
)
from src.utils.image_io import load_image
from src.utils.paths import repo_path
from src.vlm.qa_metrics import anls, exact_match
from src.vlm.run_vlm import run_vlm_overview_patches, run_vlm_single

VLM_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


def main() -> None:
    """CLI: VLM QA eval over a DocVQA manifest."""
    parser = argparse.ArgumentParser(description="VLM DocVQA evaluation.")
    parser.add_argument("--manifest", required=True, help="DocVQA JSONL manifest")
    parser.add_argument(
        "--method", default="bops",
        choices=["resize", "overview_only", "random", "uniform", "bops"],
    )
    parser.add_argument("--num-patches", type=int, default=4, help="Patch budget for patch modes")
    parser.add_argument("--limit", type=int, default=5, help="Max QA samples")
    parser.add_argument("--dry-run", action="store_true", help="Skip model; placeholder answers")
    parser.add_argument("--experiment-stage", default="sanity", choices=["debug", "sanity", "pilot", "paper"])
    parser.add_argument("--output", default=None, help="Override output CSV path")
    parser.add_argument("--append", action="store_true", help="Append rows to existing CSV")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output CSV")
    parser.add_argument("--run-id", default=None, help="Run identifier")
    args = parser.parse_args()

    run_id = args.run_id or new_run_id()
    mode = "DRY-RUN" if args.dry_run else "REAL"
    csv_path = Path(args.output) if args.output else default_vlm_metrics_path(
        args.manifest, args.method, args.num_patches
    )
    print(f"MODE: {mode} | method={args.method} | patches={args.num_patches}")
    print(f"experiment_stage={args.experiment_stage} | run_id={run_id}")
    print(f"Output: {csv_path}")

    rows: list[dict] = []
    for i, record in enumerate(iter_manifest(args.manifest)):
        if i >= args.limit:
            break
        image = load_image(repo_path(record["image_path"]))
        question = record.get("question", "")
        answers = record.get("answer", [])
        if isinstance(answers, str):
            answers = [answers]

        print(f"  [{i+1}/{args.limit}] {record['image_id']} | {args.method} ...", flush=True)
        t0 = time.perf_counter()
        invalid_budget = False

        if args.dry_run:
            raw_pred, parsed = "dry-run", "dry-run"
        elif args.method == "resize":
            resized, meta = resize_to_area_ratio(image, 0.25)
            invalid_budget = bool(meta.get("invalid_budget", False))
            raw_pred, parsed = run_vlm_single(resized, question)
        elif args.method == "overview_only":
            result = run_bops(image, 0, mode="overview_only")
            invalid_budget = bool(result["meta"].get("invalid_budget", False))
            raw_pred, parsed = run_vlm_single(result["overview"], question)
        else:
            mode_bops = "ocr_guided" if args.method == "bops" else args.method
            result = run_bops(image, args.num_patches, mode=mode_bops)
            invalid_budget = bool(result["meta"].get("invalid_budget", False))
            raw_pred, parsed = run_vlm_overview_patches(
                result["overview"], result["patches"], question
            )

        runtime = round(time.perf_counter() - t0, 3)
        em = exact_match(parsed, answers)
        anls_score = anls(parsed, answers)
        preview = (parsed[:60] + "…") if len(parsed) > 60 else parsed
        print(f"       -> parsed={preview!r} | EM={em} ANLS={anls_score:.3f} | {runtime}s", flush=True)

        rows.append({
            "run_id": run_id,
            "timestamp": iso_timestamp(),
            "experiment_stage": args.experiment_stage,
            "image_id": record["image_id"],
            "method": args.method,
            "num_patches": args.num_patches,
            "question": question,
            "ground_truth_answer": serialize_answers(answers),
            "raw_prediction": raw_pred,
            "parsed_prediction": parsed,
            "exact_match": em,
            "anls": anls_score,
            "runtime_sec": runtime,
            "model_name": VLM_MODEL_NAME,
            "invalid_budget": invalid_budget,
            "not_applicable": False,
            "dry_run": args.dry_run,
        })

    write_or_append_csv(rows, csv_path, append=args.append, overwrite=args.overwrite)
    print(f"Wrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
