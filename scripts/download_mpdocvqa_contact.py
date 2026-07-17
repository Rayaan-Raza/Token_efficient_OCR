#!/usr/bin/env python3
"""Build MP-DocVQA contact-sheet transfer manifests from Hugging Face or local data.

This script creates the MP-DocVQA *contact-sheet transfer setting* (not the
standard MP-DocVQA benchmark). Each document is flattened into a single
budgeted contact-sheet image, and QA pairs point to that composite. Gold page
labels (e.g. ``answer_page_idx``) are intentionally ignored for leakage safety.

Expected Hugging Face dataset id: ``lmms-lab/MP-DocVQA`` (adjust with
``--dataset-id`` if your environment uses a different id).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from itertools import islice
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from datasets import load_dataset, load_from_disk
from PIL import Image
from tqdm import tqdm

from src.data.contact_sheet import build_contact_sheet
from src.utils.image_io import save_image
from src.utils.paths import data_path, repo_path

HF_DATASET_ID = "lmms-lab/MP-DocVQA"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download MP-DocVQA subset and build contact-sheet manifests."
    )
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
        default=None,
        help="Optional dataset config name.",
    )
    p.add_argument(
        "--split",
        default="val",
        help="Dataset split to stream.",
    )
    p.add_argument(
        "--local-dataset",
        type=Path,
        default=None,
        help="Path to a dataset saved via datasets.save_to_disk.",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root for resolving relative page image paths.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Image root (default: data/raw/mpdocvqa_contact).",
    )
    p.add_argument(
        "--manifest-name",
        default="",
        help="Override manifest filename under data/manifests/.",
    )
    p.add_argument(
        "--max-side",
        type=int,
        default=2048,
        help="Maximum side length for the contact-sheet image.",
    )
    p.add_argument(
        "--padding",
        type=int,
        default=10,
        help="Padding between pages in the contact sheet.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip writing contact sheets that already exist on disk.",
    )
    return p.parse_args()


def _normalize_answers(value: object) -> list[str]:
    """Flatten HF answer fields into plain answer strings.

    Handles nested lists, numpy/pyarrow arrays, and accidental stringified
    Python lists such as ``\"['0.28']\"``.
    """
    import ast

    if value is None:
        return []

    # numpy / pyarrow array-like
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            value = value.tolist()
        except Exception:
            pass

    out: list[str] = []

    def _append(item: object) -> None:
        if item is None:
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                _append(child)
            return
        if hasattr(item, "tolist") and not isinstance(item, (str, bytes)):
            try:
                _append(item.tolist())
                return
            except Exception:
                pass
        text = str(item).strip()
        if not text:
            return
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                parsed = None
            if isinstance(parsed, (list, tuple)):
                for child in parsed:
                    _append(child)
                return
        out.append(text)

    _append(value)
    # Preserve order while dropping exact duplicates.
    seen: set[str] = set()
    deduped: list[str] = []
    for ans in out:
        if ans not in seen:
            seen.add(ans)
            deduped.append(ans)
    return deduped


def _safe_id(value: object, fallback: str) -> str:
    text = str(value) if value is not None else fallback
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def _load_image_from_path(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _extract_page_images(row: dict, data_root: Path | None) -> list[Image.Image]:
    def _load(obj: object) -> Image.Image | None:
        if isinstance(obj, Image.Image):
            return obj.convert("RGB")
        if isinstance(obj, str):
            path = Path(obj)
            if data_root is not None:
                path = data_root / path
            return _load_image_from_path(path)
        if isinstance(obj, dict):
            if "image" in obj:
                return _load(obj["image"])
            for key in ("path", "file_name", "filename", "image_path"):
                if key in obj:
                    return _load(obj[key])
        return None

    for key in ("images", "pages", "page_images", "page"):
        if key in row:
            value = row[key]
            if isinstance(value, (list, tuple)):
                images = [_load(item) for item in value]
                return [img for img in images if img is not None]
            img = _load(value)
            return [img] if img is not None else []

    if "image" in row:
        value = row["image"]
        if isinstance(value, (list, tuple)):
            images = [_load(item) for item in value]
            return [img for img in images if img is not None]
        img = _load(value)
        return [img] if img is not None else []

    image_keys = sorted(
        (k for k in row.keys() if re.match(r"^image_\d+$", str(k))),
        key=lambda k: int(str(k).split("_")[1]),
    )
    if image_keys:
        images = [_load(row[k]) for k in image_keys]
        return [img for img in images if img is not None]

    return []


def _iter_rows(dataset: Iterable, limit: int) -> Iterable:
    return islice(dataset, limit) if limit else dataset


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or data_path("raw", "mpdocvqa_contact")
    img_dir = Path(out_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = data_path("manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)

    n = int(args.num_samples)
    manifest_path = manifest_dir / (args.manifest_name or f"mpdocvqa_contact_{n}.jsonl")

    if args.local_dataset is not None:
        dataset = load_from_disk(str(args.local_dataset))
        if hasattr(dataset, "keys") and args.split in dataset.keys():
            dataset = dataset[args.split]
    else:
        dataset = load_dataset(
            args.dataset_id,
            args.dataset_config,
            split=args.split,
            streaming=True,
        )

    total = n if n else None
    doc_cache: dict[str, dict] = {}
    written = 0

    with open(manifest_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(tqdm(_iter_rows(dataset, n), total=total, desc=f"mpdocvqa_contact_{n}")):
            doc_id = (
                row.get("docId")
                or row.get("doc_id")
                or row.get("document_id")
                or row.get("documentId")
                or f"doc_{i}"
            )
            safe_doc_id = _safe_id(doc_id, f"doc_{i}")

            if safe_doc_id not in doc_cache:
                pages = _extract_page_images(row, args.data_root)
                if not pages:
                    raise KeyError(
                        "No page images found for MP-DocVQA contact-sheet transfer row."
                    )
                sheet, meta = build_contact_sheet(
                    pages,
                    max_side=args.max_side,
                    padding=args.padding,
                )
                image_name = f"mpdocvqa_contact_{safe_doc_id}.png"
                image_path = img_dir / image_name
                if not (args.skip_existing and image_path.exists()):
                    save_image(sheet, image_path)
                rel = image_path
                try:
                    rel = image_path.resolve().relative_to(repo_path().resolve())
                except ValueError:
                    pass
                doc_cache[safe_doc_id] = {
                    "image_path": str(rel).replace("\\", "/"),
                    "meta": meta,
                }

            answers = _normalize_answers(row.get("answers", row.get("answer")))
            question_id = row.get("questionId") or row.get("question_id") or i
            record = {
                "image_id": f"mpdocvqa_contact_{question_id}",
                "dataset": "MP-DocVQA-contact-sheet",
                "split": args.split,
                "image_path": doc_cache[safe_doc_id]["image_path"],
                "ocr_gt_text": "",
                "question": row.get("question", row.get("question_text", "")),
                "answer": answers,
                "metadata": {
                    "doc_id": doc_id,
                    "contact_sheet": doc_cache[safe_doc_id]["meta"],
                    "source_dataset_id": args.dataset_id,
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Saved contact sheets to: {img_dir}")
    print(f"Saved manifest to: {manifest_path}")
    print(f"Total samples: {written}")


if __name__ == "__main__":
    main()
