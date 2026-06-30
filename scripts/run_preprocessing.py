#!/usr/bin/env python3
"""Run preprocessing smoke test or batch transforms from YAML config."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing.resize import resize_to_area_ratio
from src.utils.config import load_config
from src.utils.image_io import load_image, save_image, write_metadata_csv
from src.utils.logging_utils import setup_logging
from src.utils.paths import data_path, outputs_path, repo_path


def run_smoke(cfg: dict) -> None:
    logger = setup_logging()
    img_rel = cfg["input_image"]
    img_path = data_path(*Path(img_rel).parts) if not Path(img_rel).is_absolute() else Path(img_rel)
    if not img_path.exists():
        img_path = repo_path(img_rel)
    if not img_path.exists():
        raise FileNotFoundError(f"Input image not found: {img_rel}")

    t0 = time.perf_counter()
    image = load_image(img_path)
    area_ratio = float(cfg.get("area_ratio", 0.5))
    resized, meta = resize_to_area_ratio(image, area_ratio)

    out_dir_rel = cfg.get("output_dir", "transformed_images")
    if str(out_dir_rel).startswith("outputs/"):
        out_dir_rel = str(out_dir_rel)[len("outputs/"):]
    out_dir = outputs_path(out_dir_rel)
    out_name = f"smoke_{img_path.stem}_r{area_ratio}.png"
    out_path = out_dir / out_name
    save_image(resized, out_path)

    meta.update({
        "input_path": str(img_path),
        "output_path": str(out_path),
        "runtime_sec": round(time.perf_counter() - t0, 4),
    })
    meta_csv_rel = cfg.get("metadata_csv", "metrics/smoke_test_metadata.csv")
    if str(meta_csv_rel).startswith("outputs/"):
        meta_csv_rel = str(meta_csv_rel)[len("outputs/"):]
    meta_csv = outputs_path(meta_csv_rel)
    write_metadata_csv([meta], meta_csv)
    logger.info("Saved %s", out_path)
    logger.info("Metadata %s", meta_csv)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_smoke(cfg)


if __name__ == "__main__":
    main()
