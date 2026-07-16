#!/usr/bin/env python3
"""Stream InfographicVQA validation samples from Hugging Face into local files.

Primary Hugging Face source: ``lmms-lab/DocVQA`` config ``InfographicVQA``
(validation split). Fallback: ``HuggingFaceM4/FineVision`` / ``infographic_vqa``.

Examples::

    python scripts/download_infographicvqa_hf.py --num-samples 300
    python scripts/download_infographicvqa_hf.py --num-samples 50 --skip-existing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from src.utils.paths import data_path, repo_path

HF_DATASET_ID = "lmms-lab/DocVQA"
HF_DATASET_CONFIG = "InfographicVQA"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download InfographicVQA validation subset.")
    p.add_argument(
        "--num-samples",
        type=int,
        default=300,
        help="Number of validation QA pairs to stream.",
    )
    p.add_argument(
        "--dataset-id",
        default=HF_DATASET_ID,
        help="Hugging Face dataset id.",
    )
    p.add_argument(
        "--dataset-config",
        default=HF_DATASET_CONFIG,
        help="Optional dataset config name.",
    )
    p.add_argument(
        "--split",
        default="validation",
        help="Dataset split to stream (use --split train if needed).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Image root (default: data/raw/infographicvqa).",
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


def _normalize_answers(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def _extract_image(row: dict) -> Image.Image:
    image = row.get("image")
    if isinstance(image, Image.Image):
        return image
    images = row.get("images")
    if isinstance(images, (list, tuple)) and images:
        if isinstance(images[0], Image.Image):
            return images[0]
    raise KeyError("Expected 'image' or 'images' field in InfographicVQA dataset row.")


def _extract_qa(row: dict) -> tuple[str, list[str]]:
    question = row.get("question") or row.get("question_text")
    if question:
        return question, _normalize_answers(row.get("answers", row.get("answer")))
    texts = row.get("texts")
    if isinstance(texts, (list, tuple)) and texts:
        question = texts[0]
        answers = _normalize_answers(texts[1:]) if len(texts) > 1 else []
        return str(question), answers
    return "", _normalize_answers(row.get("answers", row.get("answer")))


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or data_path("raw", "infographicvqa")
    img_dir = Path(out_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = data_path("manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)

    n = int(args.num_samples)
    manifest_path = manifest_dir / (args.manifest_name or f"infographicvqa_{n}.jsonl")

    split_name = args.split
    try:
        dataset = load_dataset(
            args.dataset_id,
            args.dataset_config,
            split=split_name,
            streaming=True,
        )
    except ValueError as exc:
        if split_name == "validation":
            split_name = "train"
            print("Split 'validation' unavailable; falling back to 'train'.")
            dataset = load_dataset(
                args.dataset_id,
                args.dataset_config,
                split=split_name,
                streaming=True,
            )
        else:
            raise exc
    dataset = dataset.take(n)

    written = 0
    with open(manifest_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(tqdm(dataset, total=n, desc=f"infographicvqa_{n}")):
            image = _extract_image(row)
            question_id = row.get("questionId") or row.get("question_id") or i
            image_name = f"infographicvqa_val_{question_id}.png"
            image_path = img_dir / image_name

            if not (args.skip_existing and image_path.exists()):
                image.convert("RGB").save(image_path)

            rel = image_path
            try:
                rel = image_path.resolve().relative_to(repo_path().resolve())
            except ValueError:
                pass

            question, answers = _extract_qa(row)
            record = {
                "image_id": f"infographicvqa_val_{question_id}",
                "dataset": "InfographicVQA",
                "split": split_name,
                "image_path": str(rel).replace("\\", "/"),
                "ocr_gt_text": "",
                "question": question,
                "answer": answers,
                "metadata": {
                    "question_id": question_id,
                    "source_dataset_id": args.dataset_id,
                    "source": row.get("source"),
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Saved images to: {img_dir}")
    print(f"Saved manifest to: {manifest_path}")
    print(f"Total samples: {written}")


if __name__ == "__main__":
    main()
