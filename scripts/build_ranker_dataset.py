#!/usr/bin/env python3
"""Build cached ranking dataset from patch labels + OCR features.

Labels are targets only; never written into the feature columns.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.data.dataset_loader import iter_manifest
from src.features.ocr_layout_graph import build_ocr_layout_graph
from src.features.ranker_features import FEATURE_KEYS, FORBIDDEN_FEATURE_COLUMNS, assert_no_feature_leakage, extract_ranker_features_for_patches
from src.preprocessing.patch_grid import Patch
from src.utils.image_io import load_image
from src.utils.logging_utils import log_progress, log_section, setup_experiment_logging
from src.utils.ocr_cache import load_cached_ocr_boxes
from src.utils.paths import outputs_path


def _manifest_tag(manifest: Path) -> str:
    name = manifest.stem
    return name.replace("docvqa_", "") if name.startswith("docvqa_") else name


def _ensure_split(image_ids: set[str], logger) -> dict:
    split_path = outputs_path("gates", "docvqa_ranker_split.json")
    if not split_path.exists():
        logger.info("Building ranker split via build_docvqa_manifests.py ...")
        import subprocess
        subprocess.check_call([sys.executable, "scripts/build_docvqa_manifests.py"], cwd=str(REPO_ROOT))
    with open(split_path, encoding="utf-8") as f:
        split = json.load(f)
    train = set(split["train_image_ids"]) & image_ids
    val = set(split["val_image_ids"]) & image_ids
    missing = image_ids - train - val
    # Any image not in the global split: assign by hash to keep deterministic
    if missing:
        logger.warning("%d images not in global split; assigning by hash", len(missing))
        for iid in sorted(missing):
            (val if hash(iid) % 5 == 0 else train).add(iid)
    return {"train_image_ids": sorted(train), "val_image_ids": sorted(val), "seed": split.get("seed", 42)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build learned-ranker feature dataset.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=outputs_path("labels", "patch_labels.parquet"))
    parser.add_argument("--split-by", default="image_id", help="Must be image_id (enforced).")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.split_by != "image_id":
        raise SystemExit("--split-by must be image_id (no question-level splits).")

    logger = setup_experiment_logging("build_ranker_dataset")
    log_section(logger, f"Ranker dataset | manifest={args.manifest}")

    if not args.labels.exists():
        logger.error("Missing labels: %s", args.labels)
        sys.exit(1)

    records = list(iter_manifest(args.manifest))
    if args.limit:
        records = records[: args.limit]

    # Index questions by image_id from manifest
    questions_by_image: dict[str, list[dict]] = {}
    for rec in records:
        iid = rec.get("image_id") or rec.get("doc_id") or rec.get("question_id", "unknown")
        questions_by_image.setdefault(iid, []).append(rec)

    labels_df = pd.read_parquet(args.labels)
    labels_df = labels_df[labels_df["image_id"].isin(questions_by_image.keys())]
    if labels_df.empty:
        logger.error("No label rows for images in manifest")
        sys.exit(1)

    split = _ensure_split(set(questions_by_image.keys()), logger)
    val_images = set(split["val_image_ids"])

    rows: list[dict] = []
    image_ids = sorted(questions_by_image.keys())
    for i, iid in enumerate(image_ids):
        if args.verbose or (i + 1) % 10 == 0 or i == 0:
            log_progress(logger, i + 1, len(image_ids), iid)

        img_labels = labels_df[labels_df["image_id"] == iid]
        if img_labels.empty:
            continue

        # Prefer question text from labels (one question per image in docvqa subsets)
        questions = questions_by_image[iid]
        # Group label rows by question_id when present
        qids = img_labels["question_id"].astype(str).unique().tolist()
        boxes = load_cached_ocr_boxes(iid) or []
        try:
            image = load_image(questions[0]["image_path"])
        except Exception as exc:
            logger.warning("Skip %s: %s", iid, exc)
            continue
        graph = build_ocr_layout_graph(boxes)
        split_name = "val" if iid in val_images else "train"

        for qid in qids:
            sub = img_labels[img_labels["question_id"].astype(str) == str(qid)]
            if sub.empty:
                continue
            question = str(sub.iloc[0].get("question", "") or next(
                (r.get("question", "") for r in questions if str(r.get("question_id", "")) == str(qid)),
                questions[0].get("question", ""),
            ))
            patches = [
                Patch(int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]), int(r.get("patch_index", r.get("index", j))))
                for j, (_, r) in enumerate(sub.iterrows())
            ]
            feat_rows = extract_ranker_features_for_patches(
                image, patches, boxes, question, layout_graph=graph
            )
            for (_, lab), feats in zip(sub.iterrows(), feat_rows):
                strict = bool(lab["label_exact_patch_ocr"]) or bool(lab["label_fullpage_box_overlap"])
                any_pos = bool(lab["label_positive"])
                group_id = f"{iid}::{qid}"
                row = {
                    "image_id": iid,
                    "question_id": str(qid),
                    "group_id": group_id,
                    "patch_index": int(lab.get("patch_index", lab.get("index", 0))),
                    "x": int(lab["x"]),
                    "y": int(lab["y"]),
                    "w": int(lab["w"]),
                    "h": int(lab["h"]),
                    "split": split_name,
                    "strict_positive": int(strict),
                    "any_positive": int(any_pos),
                    "label_confidence": float(lab.get("label_confidence", 1.0)),
                }
                row.update(feats)
                rows.append(row)

    if not rows:
        logger.error("No rows produced")
        sys.exit(1)

    df = pd.DataFrame(rows)
    assert_no_feature_leakage([c for c in df.columns if c in FEATURE_KEYS])
    leaked = set(df.columns) & FORBIDDEN_FEATURE_COLUMNS
    # Targets allowed in dataframe, but not as FEATURE_KEYS
    allowed_targets = {"strict_positive", "any_positive", "label_confidence"}
    bad = leaked - allowed_targets
    if bad:
        raise SystemExit(f"Forbidden columns leaked into dataset: {bad}")

    tag = _manifest_tag(args.manifest)
    out_dir = outputs_path("ranker")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / f"ranker_dataset_{tag}.parquet"
    df.to_parquet(out_parquet, index=False)

    schema = {
        "feature_keys": FEATURE_KEYS,
        "forbidden_feature_columns": sorted(FORBIDDEN_FEATURE_COLUMNS),
        "targets": ["strict_positive", "any_positive"],
        "n_rows": len(df),
        "n_images": int(df["image_id"].nunique()),
        "n_questions": int(df["question_id"].nunique()),
        "n_train_images": int(df.loc[df["split"] == "train", "image_id"].nunique()),
        "n_val_images": int(df.loc[df["split"] == "val", "image_id"].nunique()),
        "manifest": str(args.manifest),
        "split": split,
    }
    schema_path = out_dir / f"ranker_dataset_{tag}_schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    # Convenience symlink/copy tag for default "latest" used by trainers
    default_path = out_dir / "ranker_dataset.parquet"
    df.to_parquet(default_path, index=False)
    with open(out_dir / "ranker_dataset_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    logger.info(
        "Wrote %s (%d rows, %d images, train=%d val=%d)",
        out_parquet, len(df), schema["n_images"], schema["n_train_images"], schema["n_val_images"],
    )


if __name__ == "__main__":
    main()
