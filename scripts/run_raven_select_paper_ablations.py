#!/usr/bin/env python3
"""Generate the exact RAVEN-Select ablation table used in the method paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.answer_selection.dataset import _load_ocr_presence
from src.metrics.statistical_tests import bootstrap_ci
from src.routing import normalize_pred
from src.routing.ensembles import pick_shortest_nonempty
from src.routing.train import load_methods
from src.utils.paths import outputs_path

METHODS = ["resize", "bm25", "ler_bops"]


def _shortest(candidates: list[tuple[str, str]], default: str = "resize") -> str:
    if not candidates:
        return default
    candidates.sort(key=lambda x: (len(x[1]), 0 if x[0] == default else 1))
    return candidates[0][0]


def _select(
    preds: dict[str, str],
    flags: dict[str, dict[str, bool]],
    *,
    methods: list[str],
    presence: str,
    shortest: bool = True,
) -> str:
    """Select using page/patch/union OCR presence, then shortest or route priority."""
    nonempty = [(m, normalize_pred(preds.get(m, ""))) for m in methods if normalize_pred(preds.get(m, ""))]
    if not nonempty:
        return "resize" if "resize" in methods else methods[0]

    present: list[tuple[str, str]] = []
    for m, norm in nonempty:
        page = flags.get(m, {}).get("page", False)
        patch = flags.get(m, {}).get("patch", False)
        ok = page if presence == "page" else patch if presence == "patch" else (page or patch)
        if ok:
            present.append((m, norm))

    if present:
        if shortest:
            return _shortest(present, default="resize")
        # No conciseness rule: deterministic route priority, resize first.
        for m in methods:
            if any(pm == m for pm, _ in present):
                return m

    # No grounded output: standard shortest-nonempty fallback over available routes.
    return pick_shortest_nonempty({m: preds[m] for m in methods}, default="resize")


def _evaluate_variant(data: pd.DataFrame, ocr: pd.DataFrame, variant: str) -> dict:
    anls_vec: list[float] = []
    em_vec: list[float] = []
    route_counts = {m: 0 for m in METHODS}

    for iid in data.index:
        preds = {m: str(data.loc[iid, f"pred__{m}"]) for m in METHODS}
        flags: dict[str, dict[str, bool]] = {}
        for m in METHODS:
            row = ocr.loc[(iid, m)] if (iid, m) in ocr.index else {}
            flags[m] = {
                "page": bool(row.get("pred_in_full_ocr", False)),
                "patch": bool(row.get("pred_in_patch_ocr", False)),
            }

        if variant in METHODS:
            pick = variant
        elif variant == "shortest_nonempty":
            pick = pick_shortest_nonempty(preds, default="resize")
        elif variant == "ocr_present_no_shortest":
            pick = _select(preds, flags, methods=METHODS, presence="union", shortest=False)
        elif variant == "page_ocr_shortest":
            pick = _select(preds, flags, methods=METHODS, presence="page")
        elif variant == "patch_ocr_shortest":
            pick = _select(preds, flags, methods=METHODS, presence="patch")
        elif variant == "page_or_patch_ocr_shortest":
            pick = _select(preds, flags, methods=METHODS, presence="union")
        elif variant == "no_bm25_route":
            pick = _select(preds, flags, methods=["resize", "ler_bops"], presence="union")
        elif variant == "no_ler_route":
            pick = _select(preds, flags, methods=["resize", "bm25"], presence="union")
        else:
            raise ValueError(variant)

        route_counts[pick] += 1
        anls_vec.append(float(data.loc[iid, f"anls__{pick}"]))
        em_vec.append(float(data.loc[iid, f"em__{pick}"]))

    return {
        "selector": variant,
        "anls": float(np.mean(anls_vec)),
        "em": float(np.mean(em_vec)),
        "route_counts": route_counts,
        "anls_vec": anls_vec,
    }


def _ci_bounds(value: list[float] | None) -> tuple[float | str, float | str]:
    """Accept either [mean, low, high] or cached [low, high] CI schemas."""
    if not value:
        return "", ""
    if len(value) == 3:
        return value[1], value[2]
    if len(value) == 2:
        return value[0], value[1]
    return "", ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    args = parser.parse_args()

    data = load_methods(args.n, METHODS)
    ocr = _load_ocr_presence(args.n)
    if ocr is None:
        raise FileNotFoundError(f"Missing OCR-presence cache for n={args.n}")

    variants = [
        "resize",
        "bm25",
        "ler_bops",
        "shortest_nonempty",
        "ocr_present_no_shortest",
        "page_ocr_shortest",
        "patch_ocr_shortest",
        "page_or_patch_ocr_shortest",
        "no_bm25_route",
        "no_ler_route",
    ]
    results = [_evaluate_variant(data, ocr, v) for v in variants]
    by_name = {r["selector"]: r for r in results}
    resize_vec = by_name["resize"]["anls_vec"]
    shortest_vec = by_name["shortest_nonempty"]["anls_vec"]

    for row in results:
        row["ci95_vs_resize"] = list(
            bootstrap_ci([a - b for a, b in zip(row["anls_vec"], resize_vec)])
        )
        row["ci95_vs_shortest"] = list(
            bootstrap_ci([a - b for a, b in zip(row["anls_vec"], shortest_vec)])
        )
        del row["anls_vec"]

    # Add cached learned-selector and old-router rows for comparison.
    overview_path = outputs_path("metrics", f"raven_select_overview_n{args.n}.json")
    if overview_path.exists():
        overview = json.loads(overview_path.read_text(encoding="utf-8"))
        learned = [
            m for m in overview.get("models", [])
            if m.get("model") in {"ridge", "logistic", "lgbm_reg", "lgbm_rank"}
        ]
        if learned:
            best = max(learned, key=lambda x: float(x["anls"]))
            results.append({
                "selector": f"learned_{best['model']}",
                "anls": float(best["anls"]),
                "em": float(best["em"]),
                "route_counts": best.get("route_counts", {}),
                "ci95_vs_resize": best.get("vs_resize_ci"),
                "ci95_vs_shortest": best.get("vs_shortest_ci"),
            })

    router_path = outputs_path("metrics", f"raven_router_n{args.n}_summary.json")
    if router_path.exists():
        router = json.loads(router_path.read_text(encoding="utf-8"))
        results.append({
            "selector": "old_raven_post",
            "anls": float(router["routed_anls"]),
            "em": float(router["routed_em"]),
            "route_counts": router.get("chosen_route_counts", {}),
            "ci95_vs_resize": router.get("anls_ci95"),
            "ci95_vs_shortest": None,
        })

    out = {
        "n": args.n,
        "method": "shortest nonempty prediction grounded in page-or-patch OCR",
        "leakage_audit": (
            "OCR presence checks generated prediction strings only; gold answers, "
            "ANLS/EM labels, answer_in_gold_ocr, and oracle routes are excluded."
        ),
        "rows": results,
    }
    json_path = outputs_path("metrics", f"raven_select_paper_ablations_n{args.n}.json")
    csv_path = outputs_path("metrics", f"raven_select_paper_ablations_n{args.n}.csv")
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    csv_rows = []
    for r in results:
        resize_low, resize_high = _ci_bounds(r.get("ci95_vs_resize"))
        shortest_low, shortest_high = _ci_bounds(r.get("ci95_vs_shortest"))
        csv_rows.append({
            "selector": r["selector"],
            "anls": r["anls"],
            "em": r["em"],
            "ci_low_vs_resize": resize_low,
            "ci_high_vs_resize": resize_high,
            "ci_low_vs_shortest": shortest_low,
            "ci_high_vs_shortest": shortest_high,
        })
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
