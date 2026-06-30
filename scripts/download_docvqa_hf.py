#!/usr/bin/env python3
"""Stream DocVQA validation samples from Hugging Face into local images + JSONL.

Uses ``datasets`` streaming mode to avoid downloading the full ~9.5 GB dataset.
Exports PNG images and a unified manifest compatible with :mod:`src.data`.

Default export: 500 validation QA pairs → ``data/raw/docvqa_hf/images/`` and
``data/manifests/docvqa_val_500.jsonl``.

Run::

    python scripts/download_docvqa_hf.py
"""

from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import json

NUM_SAMPLES = 500

OUT_DIR = Path("data/raw/docvqa_hf")
IMG_DIR = OUT_DIR / "images"
MANIFEST_DIR = Path("data/manifests")

IMG_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

# Stream validation only — avoids downloading/processing the full ~9.5 GB dataset.
dataset = load_dataset(
    "HuggingFaceM4/DocumentVQA",
    split="validation",
    streaming=True,
).take(NUM_SAMPLES)

manifest_path = MANIFEST_DIR / "docvqa_val_500.jsonl"

with open(manifest_path, "w", encoding="utf-8") as f:
    for i, row in enumerate(tqdm(dataset, total=NUM_SAMPLES)):
        image = row["image"]

        question_id = row.get("questionId", i)
        image_name = f"docvqa_val_{question_id}.png"
        image_path = IMG_DIR / image_name

        # Convert to RGB to avoid mode issues
        image.convert("RGB").save(image_path)

        record = {
            "image_id": f"docvqa_val_{question_id}",
            "dataset": "DocVQA",
            "split": "validation",
            "image_path": str(image_path).replace("\\", "/"),
            "ocr_gt_text": "",
            "question": row.get("question", ""),
            "answer": row.get("answers", []),
            "answer_type": row.get("question_types", []),
            "metadata": {
                "docId": row.get("docId", None),
                "ucsf_document_id": row.get("ucsf_document_id", None),
                "ucsf_document_page_no": row.get("ucsf_document_page_no", None)
            }
        }

        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Saved images to: {IMG_DIR}")
print(f"Saved manifest to: {manifest_path}")
print(f"Total samples: {NUM_SAMPLES}")
