#!/usr/bin/env python3
"""RAVEN-Select P16 feature-group ablations + gate write.

Primary method is ocr_present_shortest when it passes P14/P15.
Also ablate the best learned model for the feature-group paper table.
"""

from __future__ import annotations

import argparse
import json

from src.answer_selection.dataset import build_long_table
from src.answer_selection.evaluate import write_select_summary
from src.answer_selection.features import FEATURE_GROUPS
from src.answer_selection.train import evaluate_selector
from src.extraction.gates import RevGateResult, write_gate_report
from src.utils.logging_utils import setup_experiment_logging
from src.utils.paths import outputs_path


def _ablate(n: int, model: str, long_df, metrics_tag: str) -> dict:
    full = evaluate_selector(n, model_name=model, metrics_tag=metrics_tag, long_df=long_df)
    write_select_summary(full, tag=f"{model}_full")
    rows = [{
        "drop_group": None,
        "config": "full",
        "anls": full["anls"],
        "em": full["em"],
        "beats_shortest": full["beats_shortest_nonempty"],
        "delta_vs_full": 0.0,
    }]
    for group in FEATURE_GROUPS:
        r = evaluate_selector(
            n, model_name=model, metrics_tag=metrics_tag, long_df=long_df, drop_groups=[group],
        )
        rows.append({
            "drop_group": group,
            "config": f"no_{group}",
            "anls": r["anls"],
            "em": r["em"],
            "beats_shortest": r["beats_shortest_nonempty"],
            "delta_vs_full": r["anls"] - full["anls"],
        })
    return {"model": model, "full_anls": full["anls"], "full_em": full["em"], "full": full, "rows": rows}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--model", default="raven_select_rule")
    p.add_argument("--learned-model", default="lgbm_reg")
    p.add_argument("--metrics-tag", default="")
    args = p.parse_args()

    logger = setup_experiment_logging("raven_select_ablations")
    long_df = build_long_table(args.n, metrics_tag=args.metrics_tag)

    primary = _ablate(args.n, args.model, long_df, args.metrics_tag)
    logger.info("primary=%s ANLS=%.4f", args.model, primary["full_anls"])
    for r in primary["rows"]:
        if r["drop_group"]:
            logger.info("  no_%s ANLS=%.4f delta=%.4f", r["drop_group"], r["anls"], r["delta_vs_full"])

    learned = _ablate(args.n, args.learned_model, long_df, args.metrics_tag)
    logger.info("learned=%s ANLS=%.4f", args.learned_model, learned["full_anls"])

    out = {
        "n": args.n,
        "primary": {k: v for k, v in primary.items() if k != "full"},
        "learned": {k: v for k, v in learned.items() if k != "full"},
    }
    path = outputs_path("metrics", f"raven_select_ablations_n{args.n}.json")
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    csv_path = outputs_path("metrics", f"raven_select_ablations_n{args.n}.csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        f.write("model,config,drop_group,anls,em,delta_vs_full\n")
        for block in (primary, learned):
            for r in block["rows"]:
                f.write(
                    f"{block['model']},{r['config']},{r.get('drop_group') or ''},"
                    f"{r['anls']:.6f},{r['em']:.6f},{r.get('delta_vs_full', 0.0):.6f}\n"
                )

    # P16: OCR presence and length (pred_text) must hurt when removed from primary rule.
    # Forcing on unused type/consensus must not beat the primary (or is informational).
    primary_rows = [r for r in primary["rows"] if r.get("drop_group")]
    by_g = {r["drop_group"]: r for r in primary_rows}
    ocr_hurts = bool(by_g.get("ocr_presence") and by_g["ocr_presence"]["anls"] < primary["full_anls"] - 1e-6)
    len_hurts = bool(by_g.get("pred_text") and by_g["pred_text"]["anls"] < primary["full_anls"] - 1e-6)
    # Forcing consensus/type on should not significantly beat primary (allow tiny noise)
    force_on_ok = True
    for g in ("answer_type", "consensus"):
        if g in by_g and by_g[g]["anls"] > primary["full_anls"] + 0.005:
            force_on_ok = False
    passed = ocr_hurts and len_hurts

    gate = RevGateResult(
        name="P16_feature_ablations",
        passed=passed,
        metrics={
            "primary_model": args.model,
            "primary_full_anls": primary["full_anls"],
            "primary_ablations": primary_rows,
            "ocr_presence_hurts": ocr_hurts,
            "pred_text_length_hurts": len_hurts,
            "force_on_type_consensus_within_0.005": force_on_ok,
            "learned_model": args.learned_model,
            "learned_full_anls": learned["full_anls"],
            "learned_ablations": [r for r in learned["rows"] if r.get("drop_group")],
        },
        thresholds={"ocr_presence_hurts": True, "pred_text_length_hurts": True},
        message="PASS" if passed else f"ocr_hurts={ocr_hurts} len_hurts={len_hurts}",
    )
    write_gate_report("P16_feature_ablations", gate)
    print(json.dumps({"ablations_path": str(path), "p16_passed": gate.passed, "message": gate.message}, indent=2))
    logger.info("P16 passed=%s", gate.passed)


if __name__ == "__main__":
    main()
