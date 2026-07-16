#!/usr/bin/env python3
"""Evaluate RAVEN-Select models + equal-cost baselines on DocVQA n=300/500."""

from __future__ import annotations

import argparse
import json

from src.answer_selection.dataset import build_long_table
from src.answer_selection.evaluate import (
    write_p14_gate,
    write_p15_gate,
    write_p17_gate,
    write_select_summary,
)
from src.answer_selection.ocr_presence import build_ocr_presence_cache
from src.answer_selection.train import evaluate_selector
from src.utils.logging_utils import setup_experiment_logging
from src.utils.paths import outputs_path

MODELS = ["raven_select_rule", "ocr_present_shortest", "ridge", "logistic", "lgbm_reg", "lgbm_rank"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--metrics-tag", default="")
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--rebuild-ocr", action="store_true")
    p.add_argument("--write-gates", action="store_true", help="Write P14/P15 (n=500) and P17 scale gates")
    args = p.parse_args()

    logger = setup_experiment_logging("raven_select_eval")
    suffix = f"_{args.metrics_tag}" if args.metrics_tag else ""
    ocr_path = outputs_path(
        "metrics", f"raven_select_ocr_presence_n{args.n}{suffix}.parquet"
    )
    if args.rebuild_ocr or not ocr_path.exists():
        logger.info("Building OCR presence cache...")
        build_ocr_presence_cache(args.n, metrics_tag=args.metrics_tag)

    logger.info("Building long feature table n=%d", args.n)
    long_df = build_long_table(args.n, metrics_tag=args.metrics_tag)
    long_path = outputs_path("metrics", f"raven_select_long_n{args.n}{suffix}.parquet")
    long_df.to_parquet(long_path, index=False)

    results = []
    best = None
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        logger.info("Evaluating model=%s", model)
        r = evaluate_selector(
            args.n,
            model_name=model,
            metrics_tag=args.metrics_tag,
            long_df=long_df,
        )
        path = write_select_summary(r, tag=args.metrics_tag)
        logger.info(
            "%s ANLS=%.4f EM=%.4f vs_short=%.4f (ci_lo=%.4f) vs_resize_ci_lo=%.4f",
            model,
            r["anls"],
            r["em"],
            r["vs_shortest_nonempty"]["delta"],
            r["vs_shortest_nonempty"]["ci95"][0],
            r["vs_resize"]["ci95"][0],
        )
        results.append({
            "model": model,
            "anls": r["anls"],
            "em": r["em"],
            "beats_shortest": r["beats_shortest_nonempty"],
            "beats_resize": r["beats_resize"],
            "vs_shortest_ci": r["vs_shortest_nonempty"]["ci95"],
            "vs_resize_ci": r["vs_resize"]["ci95"],
            "route_counts": r["route_counts"],
            "baselines": r["baselines"],
            "summary_path": path,
            "full": r,
        })
        if best is None or r["anls"] > best["anls"]:
            best = r

    # Main comparison table
    rows = []
    if best:
        for name, b in best["baselines"].items():
            rows.append({"name": name, "anls": round(b["anls"], 4), "em": round(b["em"], 4), "kind": "baseline"})
        for r in results:
            rows.append({
                "name": f"raven_select_{r['model']}",
                "anls": round(r["anls"], 4),
                "em": round(r["em"], 4),
                "kind": "learned",
            })
    table_path = outputs_path(
        "metrics", f"raven_select_main_table_n{args.n}{suffix}.csv"
    )
    with table_path.open("w", encoding="utf-8", newline="") as f:
        f.write("name,anls,em,kind\n")
        for row in rows:
            f.write(f"{row['name']},{row['anls']},{row['em']},{row['kind']}\n")

    overview = {
        "n": args.n,
        "method": (best or {}).get("method") if best else None,
        "method_version": (best or {}).get("method_version") if best else None,
        "best_model": best["model"] if best else None,
        "best_anls": best["anls"] if best else None,
        "best_em": best["em"] if best else None,
        "models": [{k: v for k, v in r.items() if k != "full"} for r in results],
        "table": str(table_path),
    }
    overview_path = outputs_path(
        "metrics", f"raven_select_overview_n{args.n}{suffix}.json"
    )
    overview_path.write_text(json.dumps(overview, indent=2), encoding="utf-8")

    # Prefer the model that passes P14+P15; else highest ANLS
    gate_winner = None
    for r in results:
        full = r["full"]
        if full.get("p14_pass") and full.get("p15_pass"):
            if gate_winner is None or full["anls"] > gate_winner["anls"]:
                gate_winner = full
    if gate_winner is None:
        gate_winner = best

    if args.write_gates and gate_winner:
        if args.n == 500:
            write_p14_gate(gate_winner)
            write_p15_gate(gate_winner)
            overview["gate_model"] = gate_winner["model"]
            overview["gate_anls"] = gate_winner["anls"]
            overview["p14_pass"] = gate_winner.get("p14_pass")
            overview["p15_pass"] = gate_winner.get("p15_pass")
        p17 = write_p17_gate(gate_winner, n=args.n)
        overview["p17_status"] = p17.metrics.get("status")
        overview["p17_pass"] = p17.passed
        overview_path.write_text(json.dumps(overview, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in overview.items()}, indent=2))
    logger.info(
        "Best=%s ANLS=%.4f | gate=%s ANLS=%.4f P14=%s P15=%s",
        overview["best_model"],
        overview["best_anls"] or 0,
        overview.get("gate_model"),
        overview.get("gate_anls") or 0,
        overview.get("p14_pass"),
        overview.get("p15_pass"),
    )


if __name__ == "__main__":
    main()
