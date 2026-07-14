#!/usr/bin/env python3
"""Sweep MMR lambda for QE-BOPS evidence coverage@2."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.data.dataset_loader import iter_manifest
from src.metrics.coverage_eval import eval_selection
from src.metrics.answer_coverage import rank_candidates_by_score
from src.preprocessing.selectors import select_patches
from src.utils.image_io import load_image
from src.utils.logging_utils import log_section, setup_experiment_logging
from src.utils.ocr_cache import load_cached_ocr_boxes
from src.utils.paths import outputs_path


def _image_id(record: dict) -> str:
    return record.get("image_id") or record.get("doc_id") or record.get("question_id", "unknown")


def _answers(record: dict) -> list[str]:
    ans = record.get("answers") or record.get("answer") or []
    if isinstance(ans, str):
        return [ans]
    return list(ans)


def _patch_texts(iid: str, indices: set[int]) -> list[str]:
    path = outputs_path("ocr", "patches", f"{iid}.json")
    if not path.exists():
        return []
    import json
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [p.get("text", "") for p in data["patches"] if p.get("index") in indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="MMR lambda sweep for QE-BOPS.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--method", default="qe_bops")
    parser.add_argument("--lambdas", default="0.0,0.25,0.5,0.75,0.9,1.0")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    logger = setup_experiment_logging("mmr_sweep")
    lambdas = [float(x) for x in args.lambdas.split(",")]
    log_section(logger, f"MMR sweep | method={args.method} | K={args.k}")

    labels_df = pd.read_parquet(outputs_path("labels", "patch_labels.parquet"))
    records = list(iter_manifest(args.manifest))
    if args.limit:
        records = records[: args.limit]

    rows = []
    for lam in lambdas:
        strict_hits = any_hits = 0
        for rec in records:
            iid = _image_id(rec)
            image = load_image(rec["image_path"])
            boxes = load_cached_ocr_boxes(iid) or []
            labels = [r.to_dict() for _, r in labels_df[labels_df["image_id"] == iid].iterrows()]
            sel = select_patches(
                image, args.method, args.k, rec.get("question", ""), boxes, mmr_lambda=lam,
            )
            sel_idx = {p.index for p in sel.patches}
            ranked = rank_candidates_by_score(sel.candidate_pool, sel.meta.get("candidate_scores", []))
            ev = eval_selection(labels, sel_idx, _answers(rec), _patch_texts(iid, sel_idx), ranked)
            strict_hits += int(ev["evidence_strict"])
            any_hits += int(ev["evidence_any"])
        n = len(records)
        row = {
            "method": args.method,
            "mmr_lambda": lam,
            "k": args.k,
            "evidence_strict": strict_hits / n if n else 0.0,
            "evidence_any": any_hits / n if n else 0.0,
            "n": n,
        }
        rows.append(row)
        logger.info("lambda=%.2f strict=%.3f any=%.3f", lam, row["evidence_strict"], row["evidence_any"])

    out = outputs_path("metrics", "mmr_sweep.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
