#!/usr/bin/env python3
"""Run OCR evaluation on a TextOCR manifest (Phases 4–5, 8).

Applies preprocessing methods (original, resize, JPEG, WebP, BOPS) under declared
budgets, runs OCR (or dry-run), and writes per-sample CER/WER/word recall to
``outputs/metrics/ocr_metrics.csv``. Budget fields are included for fairness filtering.

Example::

    python scripts/run_ocr_eval.py \\
        --manifest data/manifests/textocr_pilot.jsonl \\
        --methods original resize jpeg bops \\
        --budgets area_0.5 kb_200 --limit 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset_loader import iter_manifest
from src.ocr.ocr_metrics import cer, wer, word_recall
from src.ocr.run_ocr import run_ocr_on_image
from src.preprocessing.compression import compress_image_to_file
from src.preprocessing.resize import resize_to_area_ratio
from src.preprocessing.bops import run_bops
from src.utils.image_io import load_image, write_metadata_csv
from src.utils.paths import outputs_path, repo_path


def apply_method(image, method: str, budget: str, out_dir: Path, image_id: str, fast: bool = False):
    """Apply one preprocessing method and write transformed image to disk.

    Args:
        image: Source PIL image.
        method: ``original``, ``resize``, ``jpeg``, ``webp``, or ``bops``.
        budget: Budget token (e.g. ``area_0.5``, ``kb_200``, ``patches_4``).
        out_dir: Directory for transformed outputs.
        image_id: Sample id for filenames.
        fast: If True, BOPS uses random patches (skips OCR for scoring).

    Returns:
        Tuple of (output_path, metadata dict).

    Raises:
        ValueError: Unknown method name.
    """
    meta = {"method": method, "budget": budget}
    if method == "original":
        path = out_dir / f"{image_id}_orig.png"
        image.save(path)
        meta["output_path"] = str(path)
        return path, meta
    if method == "resize":
        ratio = float(budget.replace("area_", ""))
        resized, m = resize_to_area_ratio(image, ratio)
        path = out_dir / f"{image_id}_resize_{budget}.png"
        resized.save(path)
        meta.update(m)
        meta["output_path"] = str(path)
        return path, meta
    if method in ("jpeg", "webp"):
        target = int(budget.replace("kb_", "")) * 1024
        fmt = "JPEG" if method == "jpeg" else "WEBP"
        path = out_dir / f"{image_id}_{method}_{budget}.jpg"
        m = compress_image_to_file(image, path, target, fmt=fmt)
        meta.update(m)
        return path, meta
    if method == "bops":
        if budget.startswith("patches_"):
            k = int(budget.replace("patches_", ""))
        else:
            k = 4
        mode = "random" if fast else "ocr_guided"
        result = run_bops(image, num_patches=k, mode=mode)
        path = out_dir / f"{image_id}_bops_{budget}.png"
        result["overview"].save(path)
        meta.update(result["meta"])
        meta["output_path"] = str(path)
        return path, meta
    raise ValueError(method)


def main() -> None:
    """CLI: run OCR eval over manifest × methods × budgets."""
    parser = argparse.ArgumentParser(description="OCR evaluation on a manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--methods", nargs="+", default=["original", "resize"])
    parser.add_argument("--budgets", nargs="+", default=["area_0.5"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Skip OCR; use empty predictions")
    args = parser.parse_args()

    out_dir = outputs_path("ocr_results")
    rows = []
    for i, record in enumerate(iter_manifest(args.manifest)):
        if args.limit and i >= args.limit:
            break
        img_path = repo_path(record["image_path"])
        image = load_image(img_path)
        gt = record.get("ocr_gt_text", "")
        for method in args.methods:
            for budget in args.budgets:
                t0 = time.perf_counter()
                try:
                    out_path, meta = apply_method(
                        image, method, budget, out_dir, record["image_id"], fast=args.dry_run
                    )
                    pred = "" if args.dry_run else run_ocr_on_image(out_path)
                    row = {
                        "image_id": record["image_id"],
                        "method": method,
                        "budget": budget,
                        "cer": cer(pred, gt),
                        "wer": wer(pred, gt),
                        "word_recall": word_recall(pred, gt),
                        "runtime_sec": round(time.perf_counter() - t0, 3),
                        **{k: meta.get(k) for k in ("invalid_budget", "budget_type", "budget_target", "budget_actual")},
                    }
                except Exception as e:
                    row = {"image_id": record["image_id"], "method": method, "budget": budget, "error": str(e)}
                rows.append(row)

    csv_path = outputs_path("metrics", "ocr_metrics.csv")
    write_metadata_csv(rows, csv_path)
    print(f"Wrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
