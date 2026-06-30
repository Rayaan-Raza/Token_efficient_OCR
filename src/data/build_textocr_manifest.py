"""Build TextOCR JSONL manifests from derived index files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import data_path, ensure_dir


def resolve_image_path(file_name: str) -> str:
    stem = Path(file_name).name
    rel = f"data/train_val_images/train_images/{stem}"
    return rel.replace("\\", "/")


def build_manifest(out_path: Path, limit: int | None = None, skip_missing: bool = True) -> int:
    index_path = data_path("raw", "textocr", "textocr_imgs_index.json")
    text_path = data_path("raw", "textocr", "textocr_img_text.json")
    with open(index_path, encoding="utf-8") as f:
        imgs = json.load(f)
    with open(text_path, encoding="utf-8") as f:
        img_text = json.load(f)

    ensure_dir(out_path.parent)
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for img_id, info in imgs.items():
            if limit is not None and count >= limit:
                break
            rel_path = resolve_image_path(info["file_name"])
            abs_path = REPO_ROOT / rel_path
            if skip_missing and not abs_path.exists():
                continue
            record = {
                "image_id": f"textocr_{img_id}",
                "dataset": "TextOCR",
                "split": info.get("set", "train"),
                "image_path": rel_path,
                "ocr_gt_text": img_text.get(img_id, ""),
                "question": "",
                "answer": "",
                "answer_type": "",
                "boxes": [],
                "metadata": {"width": info.get("width"), "height": info.get("height")},
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    n = build_manifest(Path(args.out), limit=args.limit)
    print(f"Wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
