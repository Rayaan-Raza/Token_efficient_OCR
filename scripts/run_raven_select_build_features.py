#!/usr/bin/env python3
"""Build RAVEN-Select feature caches (OCR presence + optional long table)."""

from __future__ import annotations

import argparse
import json

from src.answer_selection.dataset import build_long_table
from src.answer_selection.ocr_presence import build_ocr_presence_cache
from src.utils.logging_utils import setup_experiment_logging
from src.utils.paths import outputs_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--metrics-tag", default="")
    p.add_argument("--skip-long-table", action="store_true")
    args = p.parse_args()

    logger = setup_experiment_logging("raven_select_features")
    ocr_path = build_ocr_presence_cache(args.n, metrics_tag=args.metrics_tag)
    logger.info("OCR presence cache: %s", ocr_path)

    if not args.skip_long_table:
        long_df = build_long_table(args.n, metrics_tag=args.metrics_tag)
        suffix = f"_{args.metrics_tag}" if args.metrics_tag else ""
        out = outputs_path(
            "metrics", f"raven_select_long_n{args.n}{suffix}.parquet"
        )
        long_df.to_parquet(out, index=False)
        logger.info("Long table %s rows -> %s", len(long_df), out)
        print(json.dumps({"ocr_presence": str(ocr_path), "long_table": str(out), "rows": len(long_df)}, indent=2))
    else:
        print(json.dumps({"ocr_presence": str(ocr_path)}, indent=2))


if __name__ == "__main__":
    main()
