#!/usr/bin/env python3
"""Run OCR evaluation on a TextOCR manifest (Phases 4–5, 8).

Applies preprocessing methods under declared budgets, runs OCR (or dry-run),
and writes per-sample metrics. Incompatible method×budget pairs are flagged
``not_applicable=true`` and excluded from aggregates.

Example::

    python scripts/run_ocr_eval.py \\
        --manifest data/manifests/textocr_debug.jsonl \\
        --methods original resize jpeg webp \\
        --budgets area_1.0 area_0.5 area_0.25 kb_500 kb_200 \\
        --limit 10 --experiment-stage sanity
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

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
    default_ocr_metrics_path,
    iso_timestamp,
    new_run_id,
    write_or_append_csv,
)
from src.utils.image_io import load_image, save_image
from src.utils.paths import outputs_path, repo_path

MODEL_NAME_OCR = "paddle_or_easyocr"


def apply_method_baseline(
    image,
    method: str,
    budget: str,
    out_dir: Path,
    image_id: str,
) -> tuple[Path | None, dict, list[Path]]:
    """Apply a non-BOPS preprocessing method and write output image.

    Returns:
        Tuple of (main image path, metadata dict, list of patch paths for OCR).
    """
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


def main() -> None:
    """CLI: OCR eval over manifest × methods × budgets."""
    parser = argparse.ArgumentParser(description="OCR evaluation on a manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--methods", nargs="+", default=["original", "resize"])
    parser.add_argument("--budgets", nargs="+", default=["area_0.5"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Skip OCR; empty predictions")
    parser.add_argument("--experiment-stage", default="sanity", choices=["debug", "sanity", "pilot", "paper"])
    parser.add_argument("--output", default=None, help="Override output CSV path")
    parser.add_argument("--append", action="store_true", help="Append rows to existing CSV")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output CSV")
    parser.add_argument("--run-id", default=None, help="Run identifier (auto-generated if omitted)")
    args = parser.parse_args()

    run_id = args.run_id or new_run_id()
    mode = "DRY-RUN" if args.dry_run else "REAL"
    print(f"MODE: {mode} | experiment_stage={args.experiment_stage} | run_id={run_id}")

    if not args.dry_run:
        print("[OCR] Warming up OCR backend (may download models on first run)...")
        get_ocr()

    out_dir = outputs_path("ocr_results")
    rows: list[dict] = []
    skipped_na = 0

    for i, record in enumerate(iter_manifest(args.manifest)):
        if args.limit and i >= args.limit:
            break
        image_id = record["image_id"]
        img_path = repo_path(record["image_path"])
        image = load_image(img_path)
        gt = record.get("ocr_gt_text", "")

        for method in args.methods:
            for budget in args.budgets:
                if not is_ocr_budget_applicable(method, budget):
                    skipped_na += 1
                    rows.append({
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
                    print(f"  [skip NA] {image_id} {method} {budget}")
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
                        **{k: meta.get(k) for k in ("invalid_budget", "budget_type", "budget_target", "budget_actual")},
                    }
                    print(
                        f"  [{i+1}] {image_id} {method} {budget_norm} | "
                        f"word_recall={wr:.3f} cer={row['cer']:.3f} runtime={runtime}s"
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
                    print(f"  [ERROR] {image_id} {method} {budget_norm}: {e}")
                rows.append(row)

    csv_path = Path(args.output) if args.output else default_ocr_metrics_path(args.manifest, args.experiment_stage)
    write_or_append_csv(rows, csv_path, append=args.append, overwrite=args.overwrite)
    applicable = sum(1 for r in rows if not r.get("not_applicable"))
    print(f"Wrote {len(rows)} rows ({applicable} applicable, {skipped_na} not_applicable) to {csv_path}")


if __name__ == "__main__":
    main()
