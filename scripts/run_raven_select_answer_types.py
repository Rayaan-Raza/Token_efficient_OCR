#!/usr/bin/env python3
"""Report RAVEN-Select performance by question answer-type bucket."""

from __future__ import annotations

import argparse
import json
import re

import numpy as np

from src.answer_selection.baselines import choose_baseline
from src.answer_selection.dataset import _load_ocr_presence
from src.answer_selection.method_spec import PRODUCTION_FLAGS
from src.routing.ensembles import pick_shortest_nonempty
from src.routing.train import load_methods
from src.utils.logging_utils import setup_experiment_logging
from src.utils.paths import outputs_path

METHODS = ["resize", "bm25", "ler_bops"]

_ORG_RE = re.compile(r"\b(company|organization|institution|university|school|bank|agency|department)\b", re.I)
_ADDRESS_RE = re.compile(r"\b(address|location|city|state|country|zip|postal|street|road|avenue|suite)\b", re.I)
_ID_RE = re.compile(r"\b(id|invoice|account|order|reference|ref\.|code|tracking|policy|serial)\b", re.I)


def _bucket_question(question: str) -> str:
    q = (question or "").lower()
    if any(w in q for w in ("date", "when", "year", "month", "day", "time")):
        return "date"
    if any(w in q for w in ("amount", "price", "cost", "total", "fee", "value", "budget", "expense", "$")):
        return "amount/currency"
    if any(w in q for w in ("how many", "number of", "count", "quantity")):
        return "number"
    if any(w in q for w in ("who", "name", "person", "author", "signed")):
        return "person/name"
    if _ORG_RE.search(q):
        return "organization"
    if _ADDRESS_RE.search(q) or "where" in q:
        return "address/location"
    if _ID_RE.search(q):
        return "id/code"
    return "phrase/other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--metrics-tag", default="")
    args = parser.parse_args()

    logger = setup_experiment_logging("raven_select_answer_types")
    data = load_methods(args.n, METHODS, metrics_tag=args.metrics_tag)
    ocr = _load_ocr_presence(args.n, metrics_tag=args.metrics_tag)
    if ocr is None:
        raise FileNotFoundError("Missing OCR presence cache; run run_raven_select_eval.py --rebuild-ocr")

    buckets = {}
    for iid in data.index:
        q = str(data.loc[iid, "question"])
        b = _bucket_question(q)
        buckets.setdefault(b, {
            "support": 0,
            "resize_anls": [],
            "resize_em": [],
            "shortest_anls": [],
            "shortest_em": [],
            "raven_select_anls": [],
            "raven_select_em": [],
        })
        preds = {m: str(data.loc[iid, f"pred__{m}"]) for m in METHODS}
        ocr_flags = {}
        for m in METHODS:
            key = (iid, m)
            if key in ocr.index:
                row = ocr.loc[key]
                ocr_flags[m] = bool(row.get("pred_in_full_ocr", False) or row.get("pred_in_patch_ocr", False))

        resize_anls = float(data.loc[iid, "anls__resize"])
        resize_em = float(data.loc[iid, "em__resize"])
        shortest = pick_shortest_nonempty(preds, default="resize")
        raven_pick = choose_baseline(
            "raven_select_rule",
            preds,
            question=q,
            ocr_flags=ocr_flags,
            default="resize",
            rule_flags=PRODUCTION_FLAGS,
        )

        buckets[b]["support"] += 1
        buckets[b]["resize_anls"].append(resize_anls)
        buckets[b]["resize_em"].append(resize_em)
        buckets[b]["shortest_anls"].append(float(data.loc[iid, f"anls__{shortest}"]))
        buckets[b]["shortest_em"].append(float(data.loc[iid, f"em__{shortest}"]))
        buckets[b]["raven_select_anls"].append(float(data.loc[iid, f"anls__{raven_pick}"]))
        buckets[b]["raven_select_em"].append(float(data.loc[iid, f"em__{raven_pick}"]))

    rows = []
    for b, vals in sorted(buckets.items(), key=lambda kv: kv[0]):
        rows.append({
            "bucket": b,
            "support": vals["support"],
            "resize_anls": float(np.mean(vals["resize_anls"])) if vals["resize_anls"] else 0.0,
            "resize_em": float(np.mean(vals["resize_em"])) if vals["resize_em"] else 0.0,
            "shortest_anls": float(np.mean(vals["shortest_anls"])) if vals["shortest_anls"] else 0.0,
            "shortest_em": float(np.mean(vals["shortest_em"])) if vals["shortest_em"] else 0.0,
            "raven_select_anls": float(np.mean(vals["raven_select_anls"])) if vals["raven_select_anls"] else 0.0,
            "raven_select_em": float(np.mean(vals["raven_select_em"])) if vals["raven_select_em"] else 0.0,
        })

    tag = f"_{args.metrics_tag}" if args.metrics_tag else ""
    out = {
        "n": args.n,
        "metrics_tag": args.metrics_tag,
        "buckets": rows,
    }
    json_path = outputs_path("metrics", f"raven_select_answer_types_n{args.n}{tag}.json")
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    csv_path = outputs_path("metrics", f"raven_select_answer_types_n{args.n}{tag}.csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        f.write("bucket,support,resize_anls,resize_em,shortest_anls,shortest_em,raven_select_anls,raven_select_em\n")
        for r in rows:
            f.write(
                f"{r['bucket']},{r['support']},{r['resize_anls']:.6f},{r['resize_em']:.6f},"
                f"{r['shortest_anls']:.6f},{r['shortest_em']:.6f},"
                f"{r['raven_select_anls']:.6f},{r['raven_select_em']:.6f}\n"
            )

    logger.info("Wrote %s", json_path)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
