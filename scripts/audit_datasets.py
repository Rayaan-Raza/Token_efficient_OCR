#!/usr/bin/env python3
"""Audit TextOCR and DocVQA datasets before manifest building."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import data_path, outputs_path, ensure_dir


def resolve_image_path(file_name: str) -> Path:
    # file_name like train/{id}.jpg
    stem = Path(file_name).name
    candidates = [
        data_path("train_val_images", "train_images", stem),
        data_path("raw", "textocr", "images", stem),
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def audit_textocr(index_path: Path, text_path: Path) -> dict:
    with open(index_path, encoding="utf-8") as f:
        imgs = json.load(f)
    with open(text_path, encoding="utf-8") as f:
        img_text = json.load(f)

    total = len(imgs)
    with_anns = sum(1 for k in imgs if img_text.get(k, "").strip())
    missing = []
    sample_checks = []
    for img_id, info in imgs.items():
        p = resolve_image_path(info["file_name"])
        if not p.exists():
            missing.append(img_id)

    rng = random.Random(42)
    sample_ids = rng.sample(list(imgs.keys()), min(10, total))
    sample_ok = True
    for sid in sample_ids:
        info = imgs[sid]
        p = resolve_image_path(info["file_name"])
        ok = p.exists()
        sample_checks.append({"image_id": sid, "path": str(p), "exists": ok})
        sample_ok = sample_ok and ok

    return {
        "total_images": total,
        "images_with_annotations": with_anns,
        "images_missing_on_disk": len(missing),
        "missing_sample_ids": missing[:20],
        "sample_checks_passed": sample_ok,
        "sample_checks": sample_checks,
    }


def audit_docvqa(manifest_path: Path) -> dict:
    rows = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    ids = [r["image_id"] for r in rows]
    dupes = len(ids) - len(set(ids))
    missing = []
    for r in rows:
        p = REPO_ROOT / r["image_path"].replace("/", "\\")
        if not p.exists():
            p = data_path(*Path(r["image_path"]).parts[1:]) if r["image_path"].startswith("data/") else Path(r["image_path"])
        if not Path(r["image_path"]).is_absolute():
            p = REPO_ROOT / r["image_path"]
        if not p.exists():
            missing.append(r["image_id"])

    img_dir = data_path("raw", "docvqa_hf", "images")
    on_disk = len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0

    return {
        "manifest_rows": len(rows),
        "images_on_disk": on_disk,
        "missing_images": len(missing),
        "missing_sample_ids": missing[:20],
        "duplicate_ids": dupes,
    }


def write_report(report: dict, out_dir: Path) -> None:
    ensure_dir(out_dir)
    json_path = out_dir / "dataset_audit_report.json"
    md_path = out_dir / "dataset_audit_report.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    lines = ["# Dataset Audit Report\n"]
    for key, val in report.items():
        if key == "passed":
            continue
        lines.append(f"## {key}\n```json\n{json.dumps(val, indent=2)}\n```\n")
    lines.append(f"\n**Passed:** {report['passed']}\n")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {json_path} and {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    index_path = data_path("raw", "textocr", "textocr_imgs_index.json")
    text_path = data_path("raw", "textocr", "textocr_img_text.json")
    manifest_path = data_path("manifests", "docvqa_val_500.jsonl")

    textocr = {"error": "index not found"}
    if index_path.exists() and text_path.exists():
        textocr = audit_textocr(index_path, text_path)

    docvqa = {"error": "manifest not found"}
    if manifest_path.exists():
        docvqa = audit_docvqa(manifest_path)

    missing_rate = 0.0
    if textocr.get("total_images"):
        missing_rate = textocr["images_missing_on_disk"] / textocr["total_images"]

    passed = (
        textocr.get("sample_checks_passed", False)
        and missing_rate <= 0.01
        and docvqa.get("missing_images", 999) == 0
        and docvqa.get("duplicate_ids", 1) == 0
    )

    report = {"textocr": textocr, "docvqa": docvqa, "passed": passed}
    write_report(report, outputs_path("audit"))


if __name__ == "__main__":
    main()
