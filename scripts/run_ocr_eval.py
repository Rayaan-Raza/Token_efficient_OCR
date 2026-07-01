#!/usr/bin/env python3
"""Run OCR evaluation on a TextOCR manifest (Phases 4–5, 8).

Applies preprocessing methods under declared budgets, runs OCR (or dry-run),
and writes per-sample metrics. Incompatible method×budget pairs are flagged
``not_applicable=true`` and excluded from aggregates.

Supports resumable runs: checkpoints are saved every ``--checkpoint-every`` images
(default 20). Re-run the same command after interruption to continue from the
last checkpoint (use ``--overwrite`` to start fresh).

Example::

    python scripts/run_ocr_eval.py \\
        --manifest data/manifests/textocr_pilot.jsonl \\
        --methods original resize jpeg webp bops \\
        --budgets area_1.0 area_0.5 area_0.25 patches_2 patches_4 patches_8 \\
        --limit 200 --experiment-stage pilot
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset_loader import iter_manifest
from src.ocr.merge_patch_ocr import merge_patch_ocr
from src.ocr.ocr_metrics import cer, wer, word_recall
from src.ocr.run_ocr import get_ocr, run_ocr_on_image
from src.preprocessing.bops import run_bops
from src.preprocessing.compression import compress_image_to_file
from src.preprocessing.resize import resize_to_area_ratio
from src.utils.budget_compat import is_ocr_budget_applicable, normalize_original_budget
from src.utils.experiment_io import (
    default_ocr_checkpoint_path,
    default_ocr_metrics_path,
    iso_timestamp,
    load_ocr_checkpoint,
    new_run_id,
    save_ocr_checkpoint,
)
from src.utils.image_io import load_image
from src.utils.paths import outputs_path, repo_path

MODEL_NAME_OCR = "paddle_or_easyocr"
CHECKPOINT_EVERY_DEFAULT = 20

_META_KEYS = (
    "invalid_budget",
    "budget_type",
    "budget_target",
    "budget_actual",
    "byte_utilization",
    "underutilized_budget",
    "overview_pixels",
    "total_bops_pixels",
)


def _banner(title: str, lines: list[str]) -> None:
    """Print a visible section banner."""
    width = max(len(title), max((len(l) for l in lines), default=0)) + 4
    print("\n" + "=" * width, flush=True)
    print(f"  {title}", flush=True)
    print("-" * width, flush=True)
    for line in lines:
        print(f"  {line}", flush=True)
    print("=" * width + "\n", flush=True)


def _progress_bar(done: int, total: int, width: int = 30) -> str:
    """ASCII progress bar for terminal status."""
    if total <= 0:
        return "[" + "?" * width + "]"
    filled = int(width * done / total)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {done}/{total}"


def _run_fingerprint(manifest: str, methods: list[str], budgets: list[str], limit: int | None, stage: str, dry_run: bool) -> dict[str, Any]:
    """Stable config dict for checkpoint matching."""
    return {
        "manifest": str(Path(manifest).resolve()),
        "methods": sorted(methods),
        "budgets": sorted(budgets),
        "limit": limit,
        "experiment_stage": stage,
        "dry_run": dry_run,
    }


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    """Load existing metric rows from CSV."""
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_all_rows(rows: list[dict[str, Any]], path: Path) -> None:
    """Rewrite full CSV from row list."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def apply_method_baseline(
    image,
    method: str,
    budget: str,
    out_dir: Path,
    image_id: str,
) -> tuple[Path | None, dict, list[Path]]:
    """Apply a non-BOPS preprocessing method and write output image."""
    meta: dict = {"method": method, "budget": budget}
    budget = normalize_original_budget(budget)

    if method == "original":
        path = out_dir / f"{image_id}_orig.png"
        image.save(path)
        meta["output_path"] = str(path)
        return path, meta, []

    if method == "resize":
        ratio = float(budget.replace("area_", ""))
        resized, m = resize_to_area_ratio(image, ratio)
        path = out_dir / f"{image_id}_resize_{budget}.png"
        resized.save(path)
        meta.update(m)
        meta["output_path"] = str(path)
        return path, meta, []

    if method in ("jpeg", "webp"):
        target = int(budget.replace("kb_", "")) * 1024
        fmt = "JPEG" if method == "jpeg" else "WEBP"
        path = out_dir / f"{image_id}_{method}_{budget}.jpg"
        m = compress_image_to_file(image, path, target, fmt=fmt)
        meta.update(m)
        return path, meta, []

    raise ValueError(f"Unknown baseline method: {method}")


def apply_method_bops(
    image,
    budget: str,
    out_dir: Path,
    image_id: str,
    fast: bool = False,
) -> tuple[Path | None, dict, list[Path]]:
    """Run BOPS and save overview + patch images for OCR merging."""
    k = int(budget.replace("patches_", ""))
    mode = "random" if fast else "ocr_guided"
    result = run_bops(image, num_patches=k, mode=mode)

    patch_dir = out_dir / f"{image_id}_bops_{budget}"
    patch_dir.mkdir(parents=True, exist_ok=True)

    overview_path = patch_dir / "overview.png"
    result["overview"].save(overview_path)

    patch_paths: list[Path] = []
    for i, patch_img in enumerate(result["patches"]):
        p = patch_dir / f"patch_{i}.png"
        patch_img.save(p)
        patch_paths.append(p)

    meta = {
        "method": "bops",
        "budget": budget,
        "output_path": str(overview_path),
        "patch_dir": str(patch_dir),
        **result["meta"],
    }
    return overview_path, meta, patch_paths


def run_ocr_prediction(
    main_path: Path | None,
    patch_paths: list[Path],
    method: str,
    dry_run: bool,
) -> str:
    """OCR a single image or merged BOPS overview+patches."""
    if dry_run:
        return ""
    if method == "bops":
        texts: list[str] = []
        if main_path and main_path.exists():
            texts.append(run_ocr_on_image(main_path))
        for pp in patch_paths:
            texts.append(run_ocr_on_image(pp))
        return merge_patch_ocr(texts)
    if main_path:
        return run_ocr_on_image(main_path)
    return ""


def _save_checkpoint(
    *,
    checkpoint_path: Path,
    csv_path: Path,
    run_id: str,
    fingerprint: dict[str, Any],
    completed_image_ids: list[str],
    last_index: int,
    total_rows: int,
    skipped_na: int,
) -> None:
    """Flush CSV rows and update checkpoint metadata."""
    payload = {
        "run_id": run_id,
        "fingerprint": fingerprint,
        "csv_path": str(csv_path),
        "completed_image_ids": completed_image_ids,
        "last_completed_index": last_index,
        "total_rows": total_rows,
        "skipped_na": skipped_na,
        "updated_at": iso_timestamp(),
    }
    save_ocr_checkpoint(checkpoint_path, payload)
    print(
        f"\n  [CHECKPOINT] saved after image {last_index + 1} | "
        f"rows={total_rows} | file={csv_path.name}\n",
        flush=True,
    )


def main() -> None:
    """CLI: OCR eval over manifest × methods × budgets with resume support."""
    parser = argparse.ArgumentParser(description="OCR evaluation on a manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--methods", nargs="+", default=["original", "resize"])
    parser.add_argument("--budgets", nargs="+", default=["area_0.5"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Skip OCR; empty predictions")
    parser.add_argument("--experiment-stage", default="sanity", choices=["debug", "sanity", "pilot", "paper"])
    parser.add_argument("--output", default=None, help="Override output CSV path")
    parser.add_argument("--overwrite", action="store_true", help="Discard checkpoint and start fresh")
    parser.add_argument("--run-id", default=None, help="Run identifier (auto-generated if omitted)")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=CHECKPOINT_EVERY_DEFAULT,
        help="Save CSV + checkpoint every N images (default 20)",
    )
    args = parser.parse_args()

    csv_path = Path(args.output) if args.output else default_ocr_metrics_path(args.manifest, args.experiment_stage)
    checkpoint_path = default_ocr_checkpoint_path(args.manifest, args.experiment_stage)
    fingerprint = _run_fingerprint(
        args.manifest, args.methods, args.budgets, args.limit, args.experiment_stage, args.dry_run
    )
    mode = "DRY-RUN" if args.dry_run else "REAL"

    all_records = list(iter_manifest(args.manifest))
    if args.limit:
        all_records = all_records[: args.limit]
    total_images = len(all_records)

    rows: list[dict[str, Any]] = []
    completed_ids: set[str] = set()
    run_id = args.run_id or new_run_id()
    skipped_na = 0
    resumed = False

    if args.overwrite:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        if csv_path.exists():
            csv_path.unlink()
    else:
        ckpt = load_ocr_checkpoint(checkpoint_path)
        if ckpt and ckpt.get("fingerprint") == fingerprint:
            run_id = ckpt.get("run_id", run_id)
            rows = _load_csv_rows(csv_path)
            completed_ids = set(ckpt.get("completed_image_ids", []))
            skipped_na = int(ckpt.get("skipped_na", 0))
            resumed = True
        elif csv_path.exists():
            # Fallback: resume from existing CSV if checkpoint was lost
            rows = _load_csv_rows(csv_path)
            if rows and all(r.get("experiment_stage") == args.experiment_stage for r in rows[:1]):
                completed_ids = {r["image_id"] for r in rows if r.get("image_id")}
                skipped_na = sum(1 for r in rows if r.get("not_applicable") in (True, "True"))
                if completed_ids and len(completed_ids) < total_images:
                    resumed = True
                    print(
                        f"[RESUME] Recovered {len(completed_ids)} images from existing CSV "
                        f"(no checkpoint file found).\n",
                        flush=True,
                    )

    _banner(
        "OCR EVAL",
        [
            f"MODE: {mode}",
            f"stage: {args.experiment_stage} | run_id: {run_id}",
            f"manifest: {args.manifest}",
            f"images: {total_images} | methods: {', '.join(args.methods)}",
            f"budgets: {len(args.budgets)} tokens",
            f"output: {csv_path}",
            f"checkpoint every: {args.checkpoint_every} images",
            f"resume: {'YES — continuing from image ' + str(len(completed_ids) + 1) if resumed else 'NO (fresh run)'}",
        ],
    )

    if not args.dry_run:
        print("[OCR] Warming up OCR backend (may download models on first run)...", flush=True)
        get_ocr()
        print("[OCR] Backend ready.\n", flush=True)

    out_dir = outputs_path("ocr_results")
    t_run_start = time.perf_counter()
    images_done_this_session = 0

    for i, record in enumerate(all_records):
        image_id = record["image_id"]
        if image_id in completed_ids:
            continue

        print(_progress_bar(len(completed_ids), total_images), flush=True)
        print(f"  >> IMAGE {i + 1}/{total_images} | {image_id}", flush=True)

        img_path = repo_path(record["image_path"])
        image = load_image(img_path)
        gt = record.get("ocr_gt_text", "")
        image_rows: list[dict[str, Any]] = []
        image_skipped_na = 0

        for method in args.methods:
            for budget in args.budgets:
                if not is_ocr_budget_applicable(method, budget):
                    image_skipped_na += 1
                    image_rows.append({
                        "run_id": run_id,
                        "timestamp": iso_timestamp(),
                        "experiment_stage": args.experiment_stage,
                        "dry_run": args.dry_run,
                        "not_applicable": True,
                        "invalid_budget": False,
                        "image_id": image_id,
                        "method": method,
                        "budget": normalize_original_budget(budget),
                        "model_name": MODEL_NAME_OCR,
                    })
                    continue

                budget_norm = normalize_original_budget(budget)
                t0 = time.perf_counter()
                try:
                    if method == "bops":
                        main_path, meta, patch_paths = apply_method_bops(
                            image, budget_norm, out_dir, image_id, fast=args.dry_run
                        )
                    else:
                        main_path, meta, patch_paths = apply_method_baseline(
                            image, method, budget_norm, out_dir, image_id
                        )

                    pred = run_ocr_prediction(main_path, patch_paths, method, args.dry_run)
                    runtime = round(time.perf_counter() - t0, 3)
                    wr = word_recall(pred, gt)

                    row = {
                        "run_id": run_id,
                        "timestamp": iso_timestamp(),
                        "experiment_stage": args.experiment_stage,
                        "dry_run": args.dry_run,
                        "not_applicable": False,
                        "image_id": image_id,
                        "method": method,
                        "budget": budget_norm,
                        "raw_prediction": pred,
                        "ocr_nonempty": bool(pred.strip()),
                        "cer": cer(pred, gt),
                        "wer": wer(pred, gt),
                        "word_recall": wr,
                        "runtime_sec": runtime,
                        "model_name": MODEL_NAME_OCR,
                        **{k: meta.get(k) for k in _META_KEYS},
                    }
                    print(
                        f"     {method:8s} {budget_norm:12s} | "
                        f"word_recall={wr:.3f} cer={row['cer']:.3f} | {runtime:.2f}s",
                        flush=True,
                    )
                except Exception as e:
                    row = {
                        "run_id": run_id,
                        "timestamp": iso_timestamp(),
                        "experiment_stage": args.experiment_stage,
                        "dry_run": args.dry_run,
                        "not_applicable": False,
                        "image_id": image_id,
                        "method": method,
                        "budget": budget_norm,
                        "error": str(e),
                        "model_name": MODEL_NAME_OCR,
                    }
                    print(f"     [ERROR] {method} {budget_norm}: {e}", flush=True)
                image_rows.append(row)

        rows.extend(image_rows)
        skipped_na += image_skipped_na
        completed_ids.add(image_id)
        images_done_this_session += 1

        if images_done_this_session % args.checkpoint_every == 0 or len(completed_ids) == total_images:
            _write_all_rows(rows, csv_path)
            _save_checkpoint(
                checkpoint_path=checkpoint_path,
                csv_path=csv_path,
                run_id=run_id,
                fingerprint=fingerprint,
                completed_image_ids=sorted(completed_ids),
                last_index=i,
                total_rows=len(rows),
                skipped_na=skipped_na,
            )

    elapsed = round(time.perf_counter() - t_run_start, 1)
    applicable = sum(1 for r in rows if not r.get("not_applicable"))
    _banner(
        "OCR EVAL COMPLETE",
        [
            f"images processed: {len(completed_ids)}/{total_images}",
            f"rows: {len(rows)} ({applicable} applicable, {skipped_na} not_applicable)",
            f"elapsed this session: {elapsed}s",
            f"csv: {csv_path}",
            f"checkpoint: {checkpoint_path}",
        ],
    )

    if len(completed_ids) >= total_images and checkpoint_path.exists():
        checkpoint_path.unlink()


if __name__ == "__main__":
    main()
