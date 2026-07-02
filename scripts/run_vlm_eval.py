#!/usr/bin/env python3
"""Run VLM DocVQA evaluation (Phases 9–10).

Evaluates document QA under preprocessing methods. Writes per-method CSV files
(default) so comparisons are not lost across runs. Use ``merge_vlm_metrics.py``
to combine into a single table.

Supports resumable runs: checkpoints every ``--checkpoint-every`` samples
(default 10). Re-run the same command to continue; use ``--overwrite`` to start fresh.

Example::

    python scripts/run_vlm_eval.py \\
        --manifest data/manifests/docvqa_pilot.jsonl \\
        --method bops_qa --num-patches 2 --limit 100 \\
        --experiment-stage pilot --checkpoint-every 10
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset_loader import iter_manifest
from src.preprocessing.bops import run_bops
from src.preprocessing.resize import resize_to_area_ratio
from src.utils.experiment_io import (
    default_vlm_checkpoint_path,
    default_vlm_metrics_path,
    iso_timestamp,
    load_vlm_checkpoint,
    new_run_id,
    save_vlm_checkpoint,
    serialize_answers,
    write_or_append_csv,
)
from src.utils.image_io import load_image
from src.utils.paths import repo_path
from src.vlm.patch_diagnostics import compute_patch_diagnostics
from src.vlm.qa_metrics import anls, exact_match
from src.vlm.run_vlm import run_vlm_overview_patches, run_vlm_single

VLM_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
CHECKPOINT_EVERY_DEFAULT = 10

_DIAG_NA = {
    "answer_in_selected_patch_ocr": None,
    "answer_in_full_image_ocr": None,
    "num_ocr_boxes_selected": None,
    "selected_text_box_coverage": None,
    "mean_patch_score": None,
    "selected_patch_coords": None,
}


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _vlm_fingerprint(
    manifest: str,
    method: str,
    num_patches: int,
    limit: int,
    experiment_stage: str,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "manifest": str(Path(manifest).as_posix()),
        "method": method,
        "num_patches": num_patches,
        "limit": limit,
        "experiment_stage": experiment_stage,
        "dry_run": dry_run,
    }


def _flush_rows(rows: list[dict[str, Any]], csv_path: Path) -> None:
    write_or_append_csv(rows, csv_path, overwrite=True)


def _save_checkpoint(
    *,
    checkpoint_path: Path,
    csv_path: Path,
    run_id: str,
    fingerprint: dict[str, Any],
    completed_image_ids: list[str],
    last_index: int,
) -> None:
    payload = {
        "run_id": run_id,
        "fingerprint": fingerprint,
        "csv_path": str(csv_path),
        "completed_image_ids": completed_image_ids,
        "last_completed_index": last_index,
        "total_rows": len(completed_image_ids),
        "updated_at": iso_timestamp(),
    }
    save_vlm_checkpoint(checkpoint_path, payload)
    print(
        f"\n  [CHECKPOINT] saved after sample {last_index + 1} | "
        f"rows={len(completed_image_ids)} | file={csv_path.name}\n",
        flush=True,
    )


def _eval_one_sample(
    record: dict[str, Any],
    *,
    method: str,
    num_patches: int,
    dry_run: bool,
) -> dict[str, Any]:
    image_id = record["image_id"]
    image = load_image(repo_path(record["image_path"]))
    question = record.get("question", "")
    answers = record.get("answer", [])
    if isinstance(answers, str):
        answers = [answers]

    t0 = time.perf_counter()
    invalid_budget = False
    diag = dict(_DIAG_NA)
    bops_result = None
    method_label = method

    if dry_run:
        raw_pred, parsed = "dry-run", "dry-run"
    elif method == "resize":
        resized, meta = resize_to_area_ratio(image, 0.25)
        invalid_budget = bool(meta.get("invalid_budget", False))
        raw_pred, parsed = run_vlm_single(resized, question)
    elif method == "overview_only":
        bops_result = run_bops(image, 0, mode="overview_only")
        invalid_budget = bool(bops_result["meta"].get("invalid_budget", False))
        raw_pred, parsed = run_vlm_single(bops_result["overview"], question)
    else:
        if method == "bops_qa":
            mode_bops = "question_aware"
            method_label = "bops_qa"
        else:
            mode_bops = "ocr_guided" if method == "bops" else method
            method_label = method
        bops_result = run_bops(image, num_patches, mode=mode_bops, question=question)
        invalid_budget = bool(bops_result["meta"].get("invalid_budget", False))
        raw_pred, parsed = run_vlm_overview_patches(
            bops_result["overview"], bops_result["patches"], question
        )

    if bops_result is not None and not dry_run:
        diag = compute_patch_diagnostics(
            image, image_id, method_label, num_patches, bops_result, answers
        )

    runtime = round(time.perf_counter() - t0, 3)
    em = exact_match(parsed, answers)
    anls_score = anls(parsed, answers)

    return {
        "image_id": image_id,
        "method_label": method_label,
        "question": question,
        "answers": answers,
        "raw_pred": raw_pred,
        "parsed": parsed,
        "em": em,
        "anls_score": anls_score,
        "runtime": runtime,
        "invalid_budget": invalid_budget,
        "diag": diag,
        "bops_result": bops_result is not None,
    }


def main() -> None:
    """CLI: VLM QA eval over a DocVQA manifest."""
    parser = argparse.ArgumentParser(description="VLM DocVQA evaluation.")
    parser.add_argument("--manifest", required=True, help="DocVQA JSONL manifest")
    parser.add_argument(
        "--method", default="bops",
        choices=["resize", "overview_only", "random", "uniform", "bops", "bops_qa"],
    )
    parser.add_argument("--num-patches", type=int, default=4, help="Patch budget for patch modes")
    parser.add_argument("--limit", type=int, default=5, help="Max QA samples")
    parser.add_argument("--dry-run", action="store_true", help="Skip model; placeholder answers")
    parser.add_argument("--experiment-stage", default="sanity", choices=["debug", "sanity", "pilot", "paper"])
    parser.add_argument("--output", default=None, help="Override output CSV path")
    parser.add_argument("--append", action="store_true", help="Append rows (disables checkpoint resume)")
    parser.add_argument("--overwrite", action="store_true", help="Discard checkpoint and start fresh")
    parser.add_argument("--run-id", default=None, help="Run identifier")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=CHECKPOINT_EVERY_DEFAULT,
        help="Save CSV + checkpoint every N samples (default 10)",
    )
    args = parser.parse_args()

    if args.append and args.overwrite:
        parser.error("Use either --append or --overwrite, not both.")

    run_id = args.run_id or new_run_id()
    mode = "DRY-RUN" if args.dry_run else "REAL"
    csv_path = Path(args.output) if args.output else default_vlm_metrics_path(
        args.manifest, args.method, args.num_patches
    )
    checkpoint_path = default_vlm_checkpoint_path(
        args.manifest, args.method, args.num_patches, args.experiment_stage
    )
    fingerprint = _vlm_fingerprint(
        args.manifest, args.method, args.num_patches, args.limit,
        args.experiment_stage, args.dry_run,
    )

    all_records = list(iter_manifest(args.manifest))[: args.limit]
    total = len(all_records)

    rows: list[dict[str, Any]] = []
    completed_ids: set[str] = set()
    resumed = False

    use_checkpoints = not args.append

    if args.overwrite and use_checkpoints:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        if csv_path.exists():
            csv_path.unlink()
    elif use_checkpoints:
        ckpt = load_vlm_checkpoint(checkpoint_path)
        if ckpt and ckpt.get("fingerprint") == fingerprint:
            run_id = ckpt.get("run_id", run_id)
            rows = _load_csv_rows(csv_path)
            completed_ids = set(ckpt.get("completed_image_ids", []))
            resumed = True
        elif csv_path.exists():
            rows = _load_csv_rows(csv_path)
            if rows:
                completed_ids = {r["image_id"] for r in rows if r.get("image_id")}
                if completed_ids and len(completed_ids) < total:
                    resumed = True
                    print(
                        f"[RESUME] Recovered {len(completed_ids)} samples from CSV "
                        f"(no checkpoint file).\n",
                        flush=True,
                    )

    print(f"MODE: {mode} | method={args.method} | patches={args.num_patches}")
    print(f"experiment_stage={args.experiment_stage} | run_id={run_id}")
    print(f"Output: {csv_path}")
    if use_checkpoints:
        print(f"checkpoint every: {args.checkpoint_every} samples | checkpoint: {checkpoint_path.name}")
    if resumed:
        print(f"[RESUME] Continuing from {len(completed_ids)}/{total} completed samples.\n", flush=True)

    if len(completed_ids) >= total:
        print(f"Already complete ({total}/{total}). Nothing to do.", flush=True)
        if use_checkpoints and checkpoint_path.exists():
            checkpoint_path.unlink()
        return

    samples_this_session = 0
    t_run_start = time.perf_counter()

    for i, record in enumerate(all_records):
        image_id = record["image_id"]
        if image_id in completed_ids:
            continue

        print(f"  [{i+1}/{total}] {image_id} | {args.method} ...", flush=True)
        result = _eval_one_sample(
            record,
            method=args.method,
            num_patches=args.num_patches,
            dry_run=args.dry_run,
        )

        preview = (result["parsed"][:60] + "…") if len(result["parsed"]) > 60 else result["parsed"]
        if result["bops_result"]:
            print(
                f"       -> parsed={preview!r} | EM={result['em']} ANLS={result['anls_score']:.3f} | "
                f"ans_in_patch={result['diag'].get('answer_in_selected_patch_ocr')} | "
                f"{result['runtime']}s",
                flush=True,
            )
        else:
            print(
                f"       -> parsed={preview!r} | EM={result['em']} "
                f"ANLS={result['anls_score']:.3f} | {result['runtime']}s",
                flush=True,
            )

        rows.append({
            "run_id": run_id,
            "timestamp": iso_timestamp(),
            "experiment_stage": args.experiment_stage,
            "image_id": image_id,
            "method": result["method_label"],
            "num_patches": args.num_patches,
            "question": result["question"],
            "ground_truth_answer": serialize_answers(result["answers"]),
            "raw_prediction": result["raw_pred"],
            "parsed_prediction": result["parsed"],
            "exact_match": result["em"],
            "anls": result["anls_score"],
            "runtime_sec": result["runtime"],
            "model_name": VLM_MODEL_NAME,
            "invalid_budget": result["invalid_budget"],
            "not_applicable": False,
            "dry_run": args.dry_run,
            **result["diag"],
        })
        completed_ids.add(image_id)
        samples_this_session += 1

        if use_checkpoints and (
            samples_this_session % args.checkpoint_every == 0
            or len(completed_ids) == total
        ):
            _flush_rows(rows, csv_path)
            _save_checkpoint(
                checkpoint_path=checkpoint_path,
                csv_path=csv_path,
                run_id=run_id,
                fingerprint=fingerprint,
                completed_image_ids=sorted(completed_ids),
                last_index=i,
            )

    if args.append:
        write_or_append_csv(rows, csv_path, append=True, overwrite=False)
    elif not use_checkpoints:
        write_or_append_csv(rows, csv_path, overwrite=args.overwrite)

    elapsed = round(time.perf_counter() - t_run_start, 1)
    print(
        f"\nVLM EVAL {'COMPLETE' if len(completed_ids) >= total else 'PARTIAL'} | "
        f"{len(completed_ids)}/{total} samples | session={samples_this_session} new | {elapsed}s",
        flush=True,
    )
    print(f"CSV: {csv_path}", flush=True)

    if use_checkpoints and len(completed_ids) >= total and checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Checkpoint removed (run complete).", flush=True)


if __name__ == "__main__":
    main()
