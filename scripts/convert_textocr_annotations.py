#!/usr/bin/env python3
"""Convert TextOCR .txt annotations to .json and derived index files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import data_path, ensure_dir


def convert(source: Path, out_dir: Path) -> None:
    print(f"Loading {source} ...")
    with open(source, encoding="utf-8") as f:
        data = json.load(f)

    ensure_dir(out_dir)
    main_json = out_dir / "TextOCR_0.1_train.json"
    print(f"Writing {main_json} ...")
    with open(main_json, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))

    imgs_index = {}
    for img_id, info in data.get("imgs", {}).items():
        imgs_index[img_id] = {
            "file_name": info.get("file_name"),
            "width": info.get("width"),
            "height": info.get("height"),
            "set": info.get("set"),
        }

    img_to_anns = data.get("imgToAnns", {})
    ann_by_id = data.get("anns", {})
    if isinstance(ann_by_id, list):
        ann_by_id = {a["id"]: a for a in ann_by_id}

    img_text = {}
    for img_id, ann_ids in img_to_anns.items():
        texts = []
        for aid in ann_ids:
            ann = ann_by_id.get(aid)
            if ann and ann.get("utf8_string"):
                texts.append(ann["utf8_string"])
        img_text[img_id] = " ".join(texts)

    index_path = out_dir / "textocr_imgs_index.json"
    text_path = out_dir / "textocr_img_text.json"
    map_path = out_dir / "textocr_img_to_anns.json"

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(imgs_index, f, separators=(",", ":"))
    with open(text_path, "w", encoding="utf-8") as f:
        json.dump(img_text, f, separators=(",", ":"))
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(img_to_anns, f, separators=(",", ":"))

    print(f"Images: {len(imgs_index)}, with annotations: {len(img_to_anns)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=str(data_path("TextOCR_0.1_train.txt")),
        help="Source TextOCR annotation file",
    )
    parser.add_argument(
        "--out-dir",
        default=str(data_path("raw", "textocr")),
        help="Output directory",
    )
    args = parser.parse_args()
    convert(Path(args.source), Path(args.out_dir))


if __name__ == "__main__":
    main()
