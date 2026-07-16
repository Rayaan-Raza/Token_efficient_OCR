#!/usr/bin/env python3
"""Compare EasyOCR vs alternate OCR engine for RAVEN-Select grounding."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src.answer_selection.baselines import evaluate_baselines
from src.answer_selection.train import _pack_result
from src.answer_selection.ocr_presence import build_ocr_presence_cache
from src.routing.train import load_methods
from src.utils.ocr_cache import DEFAULT_OCR_ENGINE
from src.utils.logging_utils import setup_experiment_logging
from src.utils.paths import outputs_path

METHODS = ["resize", "bm25", "ler_bops"]


def _load_presence(n: int, *, metrics_tag: str, engine: str) -> tuple[str, dict]:
    path = build_ocr_presence_cache(n, metrics_tag=metrics_tag, ocr_engine=engine)
    meta_path = path.with_suffix(".json")
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return str(path), meta


def _evaluate_with_presence(data, presence_df):
    bases = evaluate_baselines(data, METHODS, ocr_presence=presence_df, default="resize")
    rule = bases["raven_select_rule"]
    return _pack_result(
        len(data),
        "raven_select_rule",
        METHODS,
        rule["anls_vec"],
        rule["em_vec"],
        rule["route_counts"],
        bases,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--metrics-tag", default="")
    parser.add_argument("--engine", default="tesseract")
    args = parser.parse_args()

    logger = setup_experiment_logging("raven_select_ocr_engine")
    data = load_methods(args.n, METHODS, metrics_tag=args.metrics_tag)

    easy_path, easy_meta = _load_presence(args.n, metrics_tag=args.metrics_tag, engine=DEFAULT_OCR_ENGINE)
    alt_path, alt_meta = _load_presence(args.n, metrics_tag=args.metrics_tag, engine=args.engine)

    if alt_meta.get("missing_full_ocr_images") == args.n:
        raise FileNotFoundError(
            f"No cached OCR boxes for engine={args.engine}; build cache before running."
        )

    easy_df = None
    alt_df = None
    try:
        easy_df = pd.read_parquet(easy_path).set_index(["image_id", "route"])
        alt_df = pd.read_parquet(alt_path).set_index(["image_id", "route"])
    except Exception as exc:
        raise RuntimeError("Failed to load OCR presence parquet") from exc

    easy = _evaluate_with_presence(data, easy_df)
    alt = _evaluate_with_presence(data, alt_df)

    anls_drop = easy["anls"] - alt["anls"]
    if anls_drop <= 0.015:
        status = "PASS"
    elif anls_drop <= 0.03:
        status = "PARTIAL"
    else:
        status = "FAIL"

    tag = f"_{args.metrics_tag}" if args.metrics_tag else ""
    out = {
        "n": args.n,
        "metrics_tag": args.metrics_tag,
        "engine": args.engine,
        "easyocr_path": easy_path,
        "alt_path": alt_path,
        "easyocr_anls": easy["anls"],
        "easyocr_em": easy["em"],
        "alt_anls": alt["anls"],
        "alt_em": alt["em"],
        "anls_drop_easy_minus_alt": anls_drop,
        "p24_status": status,
        "thresholds": {"pass": 0.015, "partial": 0.03},
    }

    out_path = outputs_path("metrics", f"raven_select_ocr_engine_n{args.n}{tag}.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    logger.info("P24 %s anls_drop=%.4f", status, anls_drop)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
