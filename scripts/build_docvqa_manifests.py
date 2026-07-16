#!/usr/bin/env python3
"""Build nested DocVQA manifests and image-level ranker splits."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.build_docvqa_manifest import build_subset
from src.utils.logging_utils import log_section, setup_experiment_logging
from src.utils.paths import data_path, outputs_path


def _unique_image_ids(rows: list[dict]) -> list[str]:
    seen = []
    for r in rows:
        iid = r.get("image_id") or r.get("doc_id") or r.get("ucsf_document_id", "")
        if iid and iid not in seen:
            seen.append(iid)
    return seen


def build_ranker_split(source_rows: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict], dict]:
    by_image: dict[str, list[dict]] = {}
    for r in source_rows:
        iid = r.get("image_id") or r.get("doc_id") or r.get("ucsf_document_id", "")
        by_image.setdefault(iid, []).append(r)
    image_ids = sorted(by_image.keys())
    rng = random.Random(seed)
    rng.shuffle(image_ids)
    n_val = max(1, int(len(image_ids) * val_fraction))
    val_ids = set(image_ids[:n_val])
    train_rows, val_rows = [], []
    for iid, rows in by_image.items():
        (val_rows if iid in val_ids else train_rows).extend(rows)
    audit = {
        "train_image_ids": sorted(set(iid for iid in image_ids if iid not in val_ids)),
        "val_image_ids": sorted(val_ids),
        "seed": seed,
        "note": "Split by image_id only; never by question_id alone.",
    }
    return train_rows, val_rows, audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Build QE-BOPS DocVQA manifests.")
    parser.add_argument("--source", type=Path, default=data_path("manifests", "docvqa_val_500.jsonl"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 300, 500])
    parser.add_argument(
        "--include-source-tag",
        action="store_true",
        help="Also write a nested manifest whose size equals the source row count.",
    )
    parser.add_argument("--ranker-val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = setup_experiment_logging("build_manifests")
    log_section(logger, "Building nested DocVQA manifests")

    source_rows = []
    with open(args.source, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                source_rows.append(json.loads(line))

    for n in args.sizes:
        out = data_path("manifests", f"docvqa_{n}.jsonl")
        build_subset(args.source, out, n)
        logger.info("Wrote %s (%d rows)", out, min(n, len(source_rows)))

    if args.include_source_tag or len(source_rows) not in set(args.sizes):
        # When scaling beyond the default sizes, also materialize the full source slice.
        n_src = len(source_rows)
        if n_src not in set(args.sizes):
            out = data_path("manifests", f"docvqa_{n_src}.jsonl")
            build_subset(args.source, out, n_src)
            logger.info("Wrote source-sized subset %s (%d rows)", out, n_src)

    train_rows, val_rows, audit = build_ranker_split(source_rows, args.ranker_val_fraction, args.seed)
    train_out = data_path("manifests", "docvqa_ranker_train.jsonl")
    val_out = data_path("manifests", "docvqa_ranker_val.jsonl")
    split_out = outputs_path("gates", "docvqa_ranker_split.json")

    for path, rows in [(train_out, train_rows), (val_out, val_rows)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("Wrote %s (%d rows, %d images)", path, len(rows), len(_unique_image_ids(rows)))

    with open(split_out, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    logger.info("Ranker split audit: %s", split_out)


if __name__ == "__main__":
    main()
