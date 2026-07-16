#!/usr/bin/env python3
"""Compute oracle headroom and RAVEN-Select recovery from cached VLM metrics."""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from src.answer_selection.train import evaluate_selector
from src.routing.train import load_methods
from src.utils.logging_utils import setup_experiment_logging
from src.utils.paths import outputs_path

METHODS = ["resize", "bm25", "ler_bops"]


def _oracle_best(data, methods: list[str]) -> dict[str, Any]:
    anls_mat = np.column_stack([data[f"anls__{m}"].to_numpy(dtype=float) for m in methods])
    em_mat = np.column_stack([data[f"em__{m}"].to_numpy(dtype=float) for m in methods])
    best_idx = anls_mat.argmax(axis=1)
    return {
        "anls": float(anls_mat.max(axis=1).mean()),
        "em": float(em_mat[np.arange(len(best_idx)), best_idx].mean()),
        "route_counts": {methods[j]: int((best_idx == j).sum()) for j in range(len(methods))},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--metrics-tag", default="")
    args = parser.parse_args()

    logger = setup_experiment_logging("raven_select_oracle")
    tag = f"_{args.metrics_tag}" if args.metrics_tag else ""

    data = load_methods(args.n, METHODS, metrics_tag=args.metrics_tag)
    oracle = _oracle_best(data, METHODS)
    resize_anls = float(data["anls__resize"].mean())
    resize_em = float(data["em__resize"].mean())

    raven = evaluate_selector(
        args.n,
        model_name="raven_select_rule",
        metrics_tag=args.metrics_tag,
    )
    raven_slim = {k: v for k, v in raven.items() if k not in ("anls_vec", "em_vec")}

    available_headroom = oracle["anls"] - resize_anls
    recovered = raven["anls"] - resize_anls
    recovery_frac = recovered / available_headroom if available_headroom > 1e-9 else 0.0

    out = {
        "n": args.n,
        "metrics_tag": args.metrics_tag,
        "methods": METHODS,
        "oracle": oracle,
        "resize": {"anls": resize_anls, "em": resize_em},
        "raven_select_rule": raven_slim,
        "available_headroom": available_headroom,
        "recovered": recovered,
        "recovery_frac": recovery_frac,
    }
    out_path = outputs_path("metrics", f"raven_select_oracle_n{args.n}{tag}.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    csv_path = outputs_path("metrics", f"raven_select_oracle_n{args.n}{tag}.csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        f.write(
            "n,metrics_tag,resize_anls,resize_em,oracle_anls,oracle_em,"
            "raven_select_anls,raven_select_em,available_headroom,recovered,recovery_frac\n"
        )
        f.write(
            f"{args.n},{args.metrics_tag},{resize_anls:.6f},{resize_em:.6f},"
            f"{oracle['anls']:.6f},{oracle['em']:.6f},{raven['anls']:.6f},"
            f"{raven['em']:.6f},{available_headroom:.6f},{recovered:.6f},"
            f"{recovery_frac:.6f}\n"
        )

    logger.info(
        "oracle=%.4f resize=%.4f raven=%.4f headroom=%.4f recovered=%.4f (%.1f%%)",
        oracle["anls"],
        resize_anls,
        raven["anls"],
        available_headroom,
        recovered,
        100 * recovery_frac,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
