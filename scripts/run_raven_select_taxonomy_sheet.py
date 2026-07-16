#!/usr/bin/env python3
"""Generate a stratified review sheet for RAVEN-Select taxonomy labeling."""

from __future__ import annotations

import argparse
import csv
import json
import random

from src.answer_selection.baselines import choose_baseline
from src.answer_selection.dataset import _load_ocr_presence
from src.answer_selection.method_spec import PRODUCTION_FLAGS
from src.answer_selection.ocr_presence import build_ocr_presence_cache
from src.routing.ensembles import pick_shortest_nonempty
from src.routing.train import load_methods
from src.utils.logging_utils import setup_experiment_logging
from src.utils.paths import outputs_path

METHODS = ["resize", "bm25", "ler_bops"]
S_LABELS = [f"S{i}" for i in range(1, 7)]
F_LABELS = [f"F{i}" for i in range(1, 9)]


def _outcome(delta: float) -> str:
    if delta > 1e-9:
        return "win"
    if delta < -1e-9:
        return "loss"
    return "tie"


def _sample_group(items: list[dict], k: int, rng: random.Random) -> list[dict]:
    if k <= 0 or not items:
        return []
    if len(items) <= k:
        return list(items)
    return rng.sample(items, k)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--metrics-tag", default="")
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = setup_experiment_logging("raven_select_taxonomy_sheet")
    data = load_methods(args.n, METHODS, metrics_tag=args.metrics_tag)
    ocr = _load_ocr_presence(args.n, metrics_tag=args.metrics_tag)
    if ocr is None:
        build_ocr_presence_cache(args.n, metrics_tag=args.metrics_tag)
        ocr = _load_ocr_presence(args.n, metrics_tag=args.metrics_tag)
    if ocr is None:
        raise FileNotFoundError("Missing OCR presence cache")

    records = []
    for iid in data.index:
        preds = {m: str(data.loc[iid, f"pred__{m}"]) for m in METHODS}
        question = str(data.loc[iid, "question"])
        gold = str(data.loc[iid, "ground_truth_answer"])
        ocr_flags = {}
        for m in METHODS:
            key = (iid, m)
            if key in ocr.index:
                row = ocr.loc[key]
                ocr_flags[m] = bool(row.get("pred_in_full_ocr", False) or row.get("pred_in_patch_ocr", False))

        shortest = pick_shortest_nonempty(preds, default="resize")
        raven = choose_baseline(
            "raven_select_rule",
            preds,
            question=question,
            ocr_flags=ocr_flags,
            default="resize",
            rule_flags=PRODUCTION_FLAGS,
        )

        resize_anls = float(data.loc[iid, "anls__resize"])
        raven_anls = float(data.loc[iid, f"anls__{raven}"])
        shortest_anls = float(data.loc[iid, f"anls__{shortest}"])
        resize_em = float(data.loc[iid, "em__resize"])
        raven_em = float(data.loc[iid, f"em__{raven}"])
        shortest_em = float(data.loc[iid, f"em__{shortest}"])

        record = {
            "image_id": iid,
            "question": question,
            "gold": gold,
            "resize_pred": preds["resize"],
            "bm25_pred": preds["bm25"],
            "ler_bops_pred": preds["ler_bops"],
            "shortest_route": shortest,
            "shortest_pred": preds[shortest],
            "raven_select_route": raven,
            "raven_select_pred": preds[raven],
            "resize_anls": resize_anls,
            "resize_em": resize_em,
            "shortest_anls": shortest_anls,
            "shortest_em": shortest_em,
            "raven_select_anls": raven_anls,
            "raven_select_em": raven_em,
            "outcome_vs_resize": _outcome(raven_anls - resize_anls),
            "outcome_vs_shortest": _outcome(raven_anls - shortest_anls),
            "route_disagree_resize": raven != "resize",
            "route_disagree_shortest": raven != shortest,
        }
        for label in S_LABELS + F_LABELS:
            record[label] = ""
        records.append(record)

    total = min(args.sample, len(records))
    rng = random.Random(args.seed)

    wins = [r for r in records if r["outcome_vs_resize"] == "win"]
    losses = [r for r in records if r["outcome_vs_resize"] == "loss"]
    ties = [r for r in records if r["outcome_vs_resize"] == "tie"]

    win_target = min(len(wins), int(round(total * 0.4)))
    loss_target = min(len(losses), int(round(total * 0.4)))
    tie_target = min(len(ties), total - win_target - loss_target)

    def _prefer_disagree(items: list[dict], k: int) -> list[dict]:
        disagree = [r for r in items if r["route_disagree_resize"] or r["route_disagree_shortest"]]
        agree = [r for r in items if r not in disagree]
        picked = _sample_group(disagree, min(k, len(disagree)), rng)
        remaining = k - len(picked)
        if remaining > 0:
            picked += _sample_group(agree, remaining, rng)
        return picked

    picked = []
    picked += _prefer_disagree(wins, win_target)
    picked += _prefer_disagree(losses, loss_target)
    picked += _prefer_disagree(ties, tie_target)

    remaining = total - len(picked)
    if remaining > 0:
        leftovers = [r for r in records if r not in picked]
        picked += _sample_group(leftovers, remaining, rng)

    tag = f"_{args.metrics_tag}" if args.metrics_tag else ""
    jsonl_path = outputs_path("labels", f"raven_select_taxonomy_review_n{args.n}{tag}.jsonl")
    csv_path = outputs_path("labels", f"raven_select_taxonomy_review_n{args.n}{tag}.csv")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    fieldnames = list(picked[0].keys()) if picked else []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in picked:
            writer.writerow(r)

    logger.info("Wrote %s and %s", jsonl_path, csv_path)
    print(json.dumps({"n": args.n, "sampled": len(picked), "jsonl": str(jsonl_path), "csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
