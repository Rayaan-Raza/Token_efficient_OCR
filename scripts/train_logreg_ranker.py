#!/usr/bin/env python3
"""Train mandatory logistic regression ranker (interpretability diagnostic)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.features.patch_features import extract_patch_features
from src.utils.image_io import load_image
from src.utils.logging_utils import log_section, setup_experiment_logging
from src.utils.ocr_cache import load_cached_ocr_boxes
from src.utils.paths import data_path, outputs_path


FEATURE_KEYS = [
    "text_coverage", "text_confidence", "edge_density", "entropy",
    "bm25", "question_overlap", "answer_type", "label_value_proximity",
    "same_row_label_value", "below_label_relation", "patch_line_density",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train logreg ranker (image-level split).")
    parser.add_argument("--labels", type=Path, default=outputs_path("labels", "patch_labels.parquet"))
    parser.add_argument("--split-by", default="image_id")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    logger = setup_experiment_logging("train_logreg")
    log_section(logger, "Logistic regression ranker (mandatory diagnostic)")

    split_path = outputs_path("gates", "docvqa_ranker_split.json")
    if not split_path.exists():
        logger.info("Building manifests/split first...")
        import subprocess
        subprocess.check_call([sys.executable, "scripts/build_docvqa_manifests.py"], cwd=str(REPO_ROOT))

    with open(split_path, encoding="utf-8") as f:
        split = json.load(f)
    val_images = set(split["val_image_ids"])

    df = pd.read_parquet(args.labels)
    if args.limit:
        df = df.head(args.limit)

    # Build feature matrix (sample subset for speed if huge)
    X_rows, y_rows, weights = [], [], []
    manifest_cache: dict[str, dict] = {}
    for _, row in df.iterrows():
        iid = row["image_id"]
        if iid not in manifest_cache:
            manifest_cache[iid] = {"boxes": load_cached_ocr_boxes(iid) or []}
        # Skip full image load in bulk — use stored patch coords + fullpage OCR boxes only
        from src.preprocessing.patch_grid import Patch
        from PIL import Image

        patch = Patch(int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"]), int(row.get("patch_index", 0)))
        # Placeholder image sized to patch bounds for edge/entropy (approximation)
        image = Image.new("RGB", (patch.x + patch.w + 1, patch.y + patch.h + 1), "white")
        feats = extract_patch_features(
            image, patch, manifest_cache[iid]["boxes"], str(row.get("question", ""))
        )
        X_rows.append([feats.get(k, 0.0) for k in FEATURE_KEYS])
        y_rows.append(int(row["label_positive"]))
        weights.append(float(row.get("label_confidence", 1.0)))

    X = np.array(X_rows, dtype=float)
    y = np.array(y_rows, dtype=int)
    w = np.array(weights, dtype=float)
    is_val = np.array([row["image_id"] in val_images for _, row in df.iterrows()])

    X_train, y_train, w_train = X[~is_val], y[~is_val], w[~is_val]
    X_val, y_val = X[is_val], y[is_val]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val) if len(X_val) else X_train_s

    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    clf.fit(X_train_s, y_train, sample_weight=w_train)

    ckpt = outputs_path("checkpoints", "logreg.pkl")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "scaler": scaler, "features": FEATURE_KEYS}, ckpt)

    coef_df = pd.DataFrame({
        "feature": FEATURE_KEYS,
        "coefficient": clf.coef_[0],
    }).sort_values("coefficient", key=abs, ascending=False)
    coef_out = outputs_path("metrics", "logreg_coefficients.csv")
    coef_df.to_csv(coef_out, index=False)

    val_acc = float((clf.predict(X_val_s) == y_val).mean()) if len(y_val) else 0.0
    logger.info("Saved %s | val_acc=%.3f | coefficients -> %s", ckpt, val_acc, coef_out)


if __name__ == "__main__":
    main()
