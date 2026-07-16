#!/usr/bin/env python3
"""Evaluate OCR-grounding normalization variants for RAVEN-Select (offline)."""

from __future__ import annotations

import argparse
import json
import re
from typing import Callable

import pandas as pd

from src.answer_selection.baselines import evaluate_baselines
from src.answer_selection.train import _pack_result
from src.metrics.answer_coverage import fuzzy_anls
from src.routing import normalize_pred
from src.routing.train import load_methods
from src.utils.logging_utils import setup_experiment_logging
from src.utils.ocr_cache import load_cached_ocr_boxes, load_cached_patch_ocr
from src.utils.paths import outputs_path

METHODS = ["resize", "bm25", "ler_bops"]
_ROUTE_TO_VLM_METHOD = {
    "resize": "resize",
    "bm25": "bm25_only",
    "ler_bops": "learned_lgbm_strict",
}
_MONTHS = {
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "may": "05",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
}


def _boxes_to_text(boxes: list[dict] | None) -> str:
    if not boxes:
        return ""
    return " ".join(str(b.get("text", "")) for b in boxes)


def _raw_norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _num_norm(text: str) -> str:
    base = normalize_pred(text)
    base = re.sub(r"\d", "0", base)
    return base


def _date_norm(text: str) -> str:
    t = (text or "").lower()
    for k, v in _MONTHS.items():
        t = re.sub(rf"\b{k}[a-z]*\b", v, t)
    t = re.sub(r"[^0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _fuzzy_match(pred: str, text: str, *, threshold: float = 0.82) -> bool:
    norm_pred = normalize_pred(pred)
    norm_text = normalize_pred(text)
    if not norm_pred or not norm_text:
        return False
    if norm_pred in norm_text:
        return True
    ptoks = norm_pred.split()
    ttoks = norm_text.split()
    if not ptoks or not ttoks:
        return False
    window = len(ptoks)
    if window > 10:
        return False
    limit = min(len(ttoks), 400)
    best = 0.0
    for i in range(max(1, limit - window + 1)):
        span = " ".join(ttoks[i : i + window])
        best = max(best, fuzzy_anls(norm_pred, span))
        if best >= threshold:
            return True
    return False


def _presence_from_cache(
    data: pd.DataFrame,
    *,
    variant: str,
    num_patches: int = 2,
) -> pd.DataFrame:
    rows = []
    for iid in data.index:
        boxes = load_cached_ocr_boxes(str(iid)) or []
        full_text = _boxes_to_text(boxes)
        for m in METHODS:
            pred = str(data.loc[iid, f"pred__{m}"])
            vlm_m = _ROUTE_TO_VLM_METHOD.get(m, m)
            patch_payload = load_cached_patch_ocr(str(iid), vlm_m, num_patches)
            patch_texts = list((patch_payload or {}).get("patch_texts", []) or [])
            patch_text = " ".join(patch_texts)

            if variant == "raw":
                pred_n = _raw_norm(pred)
                full_n = _raw_norm(full_text)
                patch_n = _raw_norm(patch_text)
                in_full = bool(pred_n and pred_n in full_n)
                in_patch = bool(pred_n and pred_n in patch_n)
            elif variant == "conservative":
                pred_n = normalize_pred(pred)
                full_n = normalize_pred(full_text)
                patch_n = normalize_pred(patch_text)
                in_full = bool(pred_n and pred_n in full_n)
                in_patch = bool(pred_n and pred_n in patch_n)
            elif variant == "num_norm":
                pred_n = _num_norm(pred)
                full_n = _num_norm(full_text)
                patch_n = _num_norm(patch_text)
                in_full = bool(pred_n and pred_n in full_n)
                in_patch = bool(pred_n and pred_n in patch_n)
            elif variant == "date_norm":
                pred_n = _date_norm(pred)
                full_n = _date_norm(full_text)
                patch_n = _date_norm(patch_text)
                in_full = bool(pred_n and pred_n in full_n)
                in_patch = bool(pred_n and pred_n in patch_n)
            elif variant == "fuzzy":
                in_full = _fuzzy_match(pred, full_text)
                in_patch = _fuzzy_match(pred, patch_text)
            else:
                raise ValueError(variant)

            rows.append({
                "image_id": iid,
                "route": m,
                "prediction": pred,
                "pred_in_full_ocr": in_full,
                "pred_in_patch_ocr": in_patch,
                "has_full_ocr": bool(full_text),
                "has_patch_ocr": bool(patch_text),
            })
    df = pd.DataFrame(rows)
    return df.set_index(["image_id", "route"])


def _evaluate_variant(
    data: pd.DataFrame,
    ocr_presence: pd.DataFrame,
    *,
    tag: str,
) -> dict:
    bases = evaluate_baselines(data, METHODS, ocr_presence=ocr_presence, default="resize")
    rule = bases["raven_select_rule"]
    packed = _pack_result(
        len(data),
        "raven_select_rule",
        METHODS,
        rule["anls_vec"],
        rule["em_vec"],
        rule["route_counts"],
        bases,
    )
    packed["normalization_variant"] = tag
    return packed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--metrics-tag", default="")
    args = parser.parse_args()

    logger = setup_experiment_logging("raven_select_normalization")
    data = load_methods(args.n, METHODS, metrics_tag=args.metrics_tag)

    variants: list[tuple[str, str]] = [
        ("raw", "RAVEN-Select-Raw"),
        ("conservative", "RAVEN-Select"),
        ("num_norm", "RAVEN-Select-NumNorm"),
        ("date_norm", "RAVEN-Select-DateNorm"),
        ("fuzzy", "RAVEN-Select-Fuzzy"),
    ]

    results = []
    for variant, name in variants:
        logger.info("Evaluating normalization=%s", variant)
        ocr_presence = _presence_from_cache(data, variant=variant)
        r = _evaluate_variant(data, ocr_presence, tag=variant)
        results.append({
            "name": name,
            "variant": variant,
            "anls": r["anls"],
            "em": r["em"],
            "route_counts": r["route_counts"],
            "vs_resize": r["vs_resize"],
            "vs_shortest_nonempty": r["vs_shortest_nonempty"],
        })

    tag = f"_{args.metrics_tag}" if args.metrics_tag else ""
    out = {
        "n": args.n,
        "metrics_tag": args.metrics_tag,
        "method": "raven_select_rule",
        "variants": results,
        "note": (
            "OCR grounding is recomputed from cached predictions + OCR boxes only; "
            "no production defaults were modified."
        ),
    }
    json_path = outputs_path("metrics", f"raven_select_normalization_n{args.n}{tag}.json")
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    csv_path = outputs_path("metrics", f"raven_select_normalization_n{args.n}{tag}.csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        f.write("name,variant,anls,em,vs_resize_delta,vs_resize_ci_low,vs_resize_ci_high\n")
        for r in results:
            vs_resize = r["vs_resize"]
            f.write(
                f"{r['name']},{r['variant']},{r['anls']:.6f},{r['em']:.6f},"
                f"{vs_resize['delta']:.6f},{vs_resize['ci95'][0]:.6f},{vs_resize['ci95'][1]:.6f}\n"
            )

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
