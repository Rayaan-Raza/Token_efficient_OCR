#!/usr/bin/env python3
"""Optional LightGBM ranker (bonus if beats heuristic)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.logging_utils import log_section, setup_experiment_logging
from src.utils.paths import outputs_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM ranker (optional).")
    parser.add_argument("--split-by", default="image_id")
    args = parser.parse_args()

    logger = setup_experiment_logging("train_lgbm")
    log_section(logger, "LightGBM ranker (optional)")

    try:
        import lightgbm as lgb
        import joblib
        import pandas as pd
        import numpy as np
        import json
        from sklearn.preprocessing import StandardScaler
        from scripts.train_logreg_ranker import FEATURE_KEYS
        from src.features.patch_features import extract_patch_features
        from src.preprocessing.patch_grid import Patch
        from PIL import Image
        from src.utils.ocr_cache import load_cached_ocr_boxes
    except ImportError:
        logger.error("lightgbm not installed — skipping optional LGBM ranker")
        sys.exit(0)

    labels = outputs_path("labels", "patch_labels.parquet")
    if not labels.exists():
        logger.error("Missing patch labels")
        sys.exit(1)

    with open(outputs_path("gates", "docvqa_ranker_split.json"), encoding="utf-8") as f:
        val_images = set(json.load(f)["val_image_ids"])

    df = pd.read_parquet(labels)
    X_rows, y_rows, w_rows, is_val = [], [], [], []
    box_cache: dict[str, list] = {}
    for _, row in df.iterrows():
        iid = row["image_id"]
        if iid not in box_cache:
            box_cache[iid] = load_cached_ocr_boxes(iid) or []
        patch = Patch(int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"]), int(row.get("patch_index", 0)))
        image = Image.new("RGB", (patch.x + patch.w + 1, patch.y + patch.h + 1), "white")
        feats = extract_patch_features(image, patch, box_cache[iid], str(row.get("question", "")))
        X_rows.append([feats.get(k, 0.0) for k in FEATURE_KEYS])
        y_rows.append(int(row["label_positive"]))
        w_rows.append(float(row.get("label_confidence", 1.0)))
        is_val.append(iid in val_images)

    X = np.array(X_rows)
    y = np.array(y_rows)
    w = np.array(w_rows)
    is_val = np.array(is_val)

    model = lgb.LGBMClassifier(
        n_estimators=300, num_leaves=31, learning_rate=0.05,
        min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    )
    model.fit(X[~is_val], y[~is_val], sample_weight=w[~is_val])
    out = outputs_path("checkpoints", "lgbm.txt")
    model.booster_.save_model(str(out))
    logger.info("Saved optional LGBM to %s", out)


if __name__ == "__main__":
    main()
