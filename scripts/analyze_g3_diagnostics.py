#!/usr/bin/env python3
"""G3 failure diagnostics: label density, random-by-seed, rank curves, error cases."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.data.dataset_loader import iter_manifest
from src.metrics.coverage_eval import eval_selection, patch_evidence_tiers, positive_patch_stats
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


def _labels_for_image(labels_df: pd.DataFrame, iid: str) -> list[dict]:
    return [r.to_dict() for _, r in labels_df[labels_df["image_id"] == iid].iterrows()]


def _patch_texts(iid: str, indices: set[int]) -> list[str]:
    path = outputs_path("ocr", "patches", f"{iid}.json")
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [p.get("text", "") for p in data["patches"] if p.get("index") in indices]


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = fieldnames or list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def analyze_label_density(records: list[dict], labels_df: pd.DataFrame) -> list[dict]:
    rows = []
    for rec in records:
        iid = _image_id(rec)
        labels = _labels_for_image(labels_df, iid)
        stats = positive_patch_stats(labels, len(labels))
        rows.append({
            "image_id": iid,
            "question": rec.get("question", ""),
            **stats,
        })
    return rows


def analyze_random_by_seed(records: list[dict], labels_df: pd.DataFrame, seeds: list[int]) -> list[dict]:
    rows = []
    for seed in seeds:
        hits_any = hits_strict = 0
        for rec in records:
            iid = _image_id(rec)
            image = load_image(rec["image_path"])
            boxes = load_cached_ocr_boxes(iid) or []
            labels = _labels_for_image(labels_df, iid)
            sel = select_patches(image, "random", 2, rec.get("question", ""), boxes, seed=seed)
            sel_idx = {p.index for p in sel.patches}
            ranked = rank_candidates_by_score(sel.candidate_pool, sel.meta.get("candidate_scores", []))
            ev = eval_selection(labels, sel_idx, _answers(rec), _patch_texts(iid, sel_idx), ranked)
            hits_any += int(ev["evidence_any"])
            hits_strict += int(ev["evidence_strict"])
        n = len(records)
        rows.append({
            "seed": seed,
            "evidence_any_at_2": hits_any / n if n else 0.0,
            "evidence_strict_at_2": hits_strict / n if n else 0.0,
            "n": n,
        })
    return rows


def analyze_rank_curves(records: list[dict], labels_df: pd.DataFrame, methods: list[str]) -> list[dict]:
    rows = []
    rank_samples: dict[str, list[float]] = {m: [] for m in methods}
    for method in methods:
        agg: dict[str, list[float]] = {}
        for rec in records:
            iid = _image_id(rec)
            image = load_image(rec["image_path"])
            boxes = load_cached_ocr_boxes(iid) or []
            labels = _labels_for_image(labels_df, iid)
            sel = select_patches(image, method, 2, rec.get("question", ""), boxes)
            sel_idx = {p.index for p in sel.patches}
            ranked = rank_candidates_by_score(sel.candidate_pool, sel.meta.get("candidate_scores", []))
            ev = eval_selection(labels, sel_idx, _answers(rec), _patch_texts(iid, sel_idx), ranked)
            for key, val in ev.items():
                if isinstance(val, bool):
                    agg.setdefault(key, []).append(float(val))
                elif val is not None:
                    agg.setdefault(key, []).append(float(val))
            r = ev.get("mean_rank_first_positive_any")
            if r is not None:
                rank_samples[method].append(r)

        n = len(records)
        row = {"method": method, "n": n}
        for key, vals in agg.items():
            row[key] = sum(vals) / len(vals) if vals else 0.0
        ranks = sorted(rank_samples[method])
        if ranks:
            row["rank_p25"] = ranks[len(ranks) // 4]
            row["rank_p50"] = ranks[len(ranks) // 2]
            row["rank_p75"] = ranks[(3 * len(ranks)) // 4]
        rows.append(row)
    return rows


def analyze_selector_errors(records: list[dict], labels_df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    rows = []
    debug_cases = []
    for rec in records:
        iid = _image_id(rec)
        image = load_image(rec["image_path"])
        boxes = load_cached_ocr_boxes(iid) or []
        labels = _labels_for_image(labels_df, iid)
        q = rec.get("question", "")

        sel_qa = select_patches(image, "bops_qa_fair_pool", 2, q, boxes)
        sel_qe = select_patches(image, "qe_bops", 2, q, boxes)
        idx_qa = {p.index for p in sel_qa.patches}
        idx_qe = {p.index for p in sel_qe.patches}
        ev_qa = eval_selection(labels, idx_qa, _answers(rec), _patch_texts(iid, idx_qa))
        ev_qe = eval_selection(labels, idx_qe, _answers(rec), _patch_texts(iid, idx_qe))

        qa_hit = ev_qa["evidence_any"]
        qe_hit = ev_qe["evidence_any"]
        row = {
            "image_id": iid,
            "question": q,
            "answers": " | ".join(_answers(rec)),
            "q_bops_hit": qa_hit,
            "qe_bops_hit": qe_hit,
            "both_hit": qa_hit and qe_hit,
            "neither_hit": not qa_hit and not qe_hit,
            "q_bops_strict": ev_qa["evidence_strict"],
            "qe_bops_strict": ev_qe["evidence_strict"],
            "q_bops_selected_text": " || ".join(_patch_texts(iid, idx_qa)),
            "qe_bops_selected_text": " || ".join(_patch_texts(iid, idx_qe)),
        }
        rows.append(row)

        if qa_hit and not qe_hit:
            debug_cases.append({
                **row,
                "case_type": "q_bops_hit_qe_miss",
                "qe_top_features": json.dumps(sel_qe.meta.get("top_features", {})),
            })

    return rows, debug_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="G3 diagnostic bundle.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    logger = setup_experiment_logging("g3_diagnostics")
    log_section(logger, "G3 failure diagnostics")

    labels_df = pd.read_parquet(outputs_path("labels", "patch_labels.parquet"))
    records = list(iter_manifest(args.manifest))
    if args.limit:
        records = records[: args.limit]

    if "-" in args.seeds:
        a, b = args.seeds.split("-")
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(x) for x in args.seeds.split(",")]

    density_rows = analyze_label_density(records, labels_df)
    _write_csv(outputs_path("metrics", "positive_label_density.csv"), density_rows)
    pos_fracs = [r["positive_patch_fraction_any"] for r in density_rows]
    pos_fracs.sort()
    logger.info(
        "Label density | mean_pos_frac=%.3f median=%.3f p75=%.3f",
        sum(pos_fracs) / len(pos_fracs),
        pos_fracs[len(pos_fracs) // 2],
        pos_fracs[(3 * len(pos_fracs)) // 4],
    )

    random_rows = analyze_random_by_seed(records, labels_df, seeds)
    _write_csv(outputs_path("metrics", "random_by_seed.csv"), random_rows)
    mean_rand = sum(r["evidence_any_at_2"] for r in random_rows) / len(random_rows)
    logger.info("Random by seed | mean evidence_any@2=%.3f (per-seed, not OR-aggregated)", mean_rand)

    rank_rows = analyze_rank_curves(
        records, labels_df, ["bops_qa_fair_pool", "qe_bops", "bops_fair_pool", "random"]
    )
    _write_csv(outputs_path("metrics", "rank_coverage_curves.csv"), rank_rows)

    error_rows, debug_cases = analyze_selector_errors(records, labels_df)
    _write_csv(outputs_path("metrics", "selector_error_table.csv"), error_rows)
    debug_dir = outputs_path("debug", "random_hits_qebops_misses")
    debug_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(debug_dir / "q_bops_hit_qe_miss.csv", debug_cases)
    with open(debug_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "q_bops_hit_qe_miss": sum(1 for r in error_rows if r["q_bops_hit"] and not r["qe_bops_hit"]),
            "qe_bops_hit_q_miss": sum(1 for r in error_rows if r["qe_bops_hit"] and not r["q_bops_hit"]),
            "both_hit": sum(1 for r in error_rows if r["both_hit"]),
            "neither_hit": sum(1 for r in error_rows if r["neither_hit"]),
        }, f, indent=2)
    logger.info("Wrote diagnostics to outputs/metrics/ and %s", debug_dir)


if __name__ == "__main__":
    main()
