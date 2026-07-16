#!/usr/bin/env python3
"""Stream DocVQA validation samples from Hugging Face into local images + JSONL.

Uses ``datasets`` streaming mode to avoid downloading the full ~9.5 GB dataset.
Exports PNG images and a unified manifest compatible with :mod:`src.data`.

Examples::

    python scripts/download_docvqa_hf.py --num-samples 500
    python scripts/download_docvqa_hf.py --num-samples 1000
    python scripts/download_docvqa_hf.py --full-validation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from src.utils.paths import data_path, repo_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download DocVQA validation subset.")
    p.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of validation QA pairs to stream (ignored with --full-validation).",
    )
    p.add_argument(
        "--full-validation",
        action="store_true",
        help="Stream the entire DocVQA validation split.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Image root (default: data/raw/docvqa_hf).",
    )
    p.add_argument(
        "--manifest-name",
        default="",
        help="Override manifest filename under data/manifests/.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip writing images that already exist on disk.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or data_path("raw", "docvqa_hf")
    img_dir = Path(out_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = data_path("manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)

    if args.full_validation:
        tag = "full"
        manifest_path = manifest_dir / (args.manifest_name or "docvqa_val_full.jsonl")
        dataset = load_dataset(
            "HuggingFaceM4/DocumentVQA",
            split="validation",
            streaming=True,
        )
        total = None
    else:
        n = int(args.num_samples)
        tag = str(n)
        manifest_path = manifest_dir / (args.manifest_name or f"docvqa_val_{n}.jsonl")
        dataset = load_dataset(
            "HuggingFaceM4/DocumentVQA",
            split="validation",
            streaming=True,
        ).take(n)
        total = n

    written = 0
    with open(manifest_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(tqdm(dataset, total=total, desc=f"docvqa_val_{tag}")):
            image = row["image"]
            question_id = row.get("questionId", i)
            image_name = f"docvqa_val_{question_id}.png"
            image_path = img_dir / image_name

            if not (args.skip_existing and image_path.exists()):
                image.convert("RGB").save(image_path)

            # Prefer the canonical Data/ path string when that tree is what loaders use.
            rel = image_path
            try:
                rel = image_path.resolve().relative_to(repo_path().resolve())
            except ValueError:
                pass

            record = {
                "image_id": f"docvqa_val_{question_id}",
                "dataset": "DocVQA",
                "split": "validation",
                "image_path": str(rel).replace("\\", "/"),
                "ocr_gt_text": "",
                "question": row.get("question", ""),
                "answer": row.get("answers", []),
                "answer_type": row.get("question_types", []),
                "metadata": {
                    "docId": row.get("docId", None),
                    "ucsf_document_id": row.get("ucsf_document_id", None),
                    "ucsf_document_page_no": row.get("ucsf_document_page_no", None),
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Saved images to: {img_dir}")
    print(f"Saved manifest to: {manifest_path}")
    print(f"Total samples: {written}")


if __name__ == "__main__":
    main()
