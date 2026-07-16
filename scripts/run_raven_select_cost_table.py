#!/usr/bin/env python3
"""Cost accounting table for RAVEN-Select (offline, cached VLM metrics)."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src.answer_selection.train import evaluate_selector
from src.routing.ensembles import pick_shortest_nonempty
from src.routing.train import METHOD_FILES, load_methods
from src.utils.logging_utils import setup_experiment_logging
from src.utils.paths import outputs_path

METHODS = ["resize", "bm25", "ler_bops"]
LEARNED_MODELS = ["ridge", "logistic", "lgbm_reg", "lgbm_rank"]

VISUAL = {
    "resize": "1 resized page",
    "bm25": "overview + K=2 patches",
    "ler_bops": "overview + K=2 patches",
    "shortest_nonempty": "all three paths",
    "raven_select_rule": "all three paths",
    "best_learned": "all three paths",
}


def _median_runtime(n: int, key: str, tag: str = "") -> float | None:
    t = f"_{tag}" if tag else ""
    path = outputs_path("metrics", METHOD_FILES[key].format(n=n, tag=t))
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "runtime_sec" not in df.columns:
        return None
    return float(df["runtime_sec"].median())


def _shortest_metrics(data: pd.DataFrame) -> tuple[float, float]:
    anls = []
    em = []
    for iid in data.index:
        preds = {m: str(data.loc[iid, f"pred__{m}"]) for m in METHODS}
        pick = pick_shortest_nonempty(preds, default="resize")
        anls.append(float(data.loc[iid, f"anls__{pick}"]))
        em.append(float(data.loc[iid, f"em__{pick}"]))
    return float(pd.Series(anls).mean()), float(pd.Series(em).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--metrics-tag", default="")
    args = parser.parse_args()

    logger = setup_experiment_logging("raven_select_cost_table")
    tag = f"_{args.metrics_tag}" if args.metrics_tag else ""

    data = load_methods(args.n, METHODS, metrics_tag=args.metrics_tag)
    runtimes = {m: _median_runtime(args.n, m, args.metrics_tag) for m in METHODS}
    sum_runtime = sum(v for v in runtimes.values() if v is not None)

    shortest_anls, shortest_em = _shortest_metrics(data)
    raven_rule = evaluate_selector(args.n, model_name="raven_select_rule", metrics_tag=args.metrics_tag)

    overview_path = outputs_path("metrics", f"raven_select_overview_n{args.n}{tag}.json")
    best_learned = None
    if overview_path.exists():
        overview = json.loads(overview_path.read_text(encoding="utf-8"))
        if overview.get("models"):
            best_learned = max(overview["models"], key=lambda r: float(r["anls"]))
    if best_learned is None:
        learned_results = [
            evaluate_selector(args.n, model_name=m, metrics_tag=args.metrics_tag)
            for m in LEARNED_MODELS
        ]
        best_learned = max(learned_results, key=lambda r: float(r["anls"]))

    rows = []
    for m in METHODS:
        rows.append({
            "method": m,
            "vlm_calls": 1,
            "visual_inputs": VISUAL[m],
            "median_runtime_sec": None if runtimes[m] is None else round(runtimes[m], 3),
            "anls": round(float(data[f"anls__{m}"].mean()), 4),
            "em": round(float(data[f"em__{m}"].mean()), 4),
            "n": args.n,
        })
    rows.append({
        "method": "shortest_nonempty",
        "vlm_calls": 3,
        "visual_inputs": VISUAL["shortest_nonempty"],
        "median_runtime_sec": round(sum_runtime, 3) if sum_runtime else None,
        "anls": round(shortest_anls, 4),
        "em": round(shortest_em, 4),
        "n": args.n,
    })
    rows.append({
        "method": "raven_select_rule",
        "vlm_calls": 3,
        "visual_inputs": VISUAL["raven_select_rule"],
        "median_runtime_sec": round(sum_runtime, 3) if sum_runtime else None,
        "anls": round(raven_rule["anls"], 4),
        "em": round(raven_rule["em"], 4),
        "n": args.n,
    })
    rows.append({
        "method": "best_learned",
        "vlm_calls": 3,
        "visual_inputs": VISUAL["best_learned"],
        "median_runtime_sec": round(sum_runtime, 3) if sum_runtime else None,
        "anls": round(float(best_learned["anls"]), 4),
        "em": round(float(best_learned["em"]), 4),
        "n": args.n,
    })

    out = {
        "n": args.n,
        "metrics_tag": args.metrics_tag,
        "rows": rows,
        "runtime_note": (
            "3-call methods use the sum of per-path median runtimes (approximate, "
            "not accounting for overlap)."
        ),
    }
    out_path = outputs_path("metrics", f"raven_select_cost_n{args.n}{tag}.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    logger.info("Wrote %s", out_path)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
