#!/usr/bin/env python3
"""Train logistic regression ranker (interpretability diagnostic).

Gate metric remains coverage@K — not AUC.
"""

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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.features.ranker_features import FEATURE_KEYS, assert_no_feature_leakage
from src.utils.logging_utils import log_section, setup_experiment_logging
from src.utils.paths import outputs_path


def _load_dataset(path: Path | None, tag: str) -> pd.DataFrame:
    candidates = []
    if path is not None and str(path) and path.exists() and path.is_file():
        candidates.append(path)
    if tag:
        candidates.append(outputs_path("ranker", f"ranker_dataset_{tag}.parquet"))
    candidates.append(outputs_path("ranker", "ranker_dataset.parquet"))
    for p in candidates:
        if p.exists() and p.is_file():
            return pd.read_parquet(p)
    raise FileNotFoundError("No ranker dataset found. Run scripts/build_ranker_dataset.py first.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train logreg ranker from cached dataset.")
    parser.add_argument("--target", choices=["strict_positive", "any_positive"], default="strict_positive")
    parser.add_argument("--from-dataset", action="store_true", default=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--dataset-tag", default="100", help="Tag like 100 or 500")
    parser.add_argument("--split-by", default="image_id")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logger = setup_experiment_logging("train_logreg")
    log_section(logger, f"Logreg ranker | target={args.target}")

    if args.split_by != "image_id":
        raise SystemExit("--split-by must be image_id")

    df = _load_dataset(args.dataset, args.dataset_tag)
    assert_no_feature_leakage(FEATURE_KEYS)
    missing = [k for k in FEATURE_KEYS if k not in df.columns]
    if missing:
        raise SystemExit(f"Dataset missing features: {missing}")

    X = df[FEATURE_KEYS].to_numpy(dtype=float)
    y = df[args.target].to_numpy(dtype=int)
    w = df["label_confidence"].to_numpy(dtype=float) if "label_confidence" in df.columns else np.ones(len(df))
    is_val = (df["split"] == "val").to_numpy()

    X_train, y_train, w_train = X[~is_val], y[~is_val], w[~is_val]
    X_val, y_val = X[is_val], y[is_val]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val) if len(X_val) else X_train_s

    # Avoid class_weight*sample_weight blow-up on rare positives
    clf = LogisticRegression(C=1.0, class_weight=None, max_iter=2000)
    # Upsample weight for positives slightly for diagnostic balance
    w_fit = w_train.copy()
    pos = y_train == 1
    if pos.any() and (~pos).any():
        w_fit[pos] *= float((~pos).sum() / max(1, pos.sum()))
    clf.fit(X_train_s, y_train, sample_weight=w_fit)

    short = "strict" if args.target == "strict_positive" else "any"
    ckpt = outputs_path("checkpoints", f"logreg_{short}.pkl")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "scaler": scaler, "features": FEATURE_KEYS, "target": args.target}, ckpt)

    coef_df = pd.DataFrame({"feature": FEATURE_KEYS, "coefficient": clf.coef_[0]}).sort_values(
        "coefficient", key=abs, ascending=False
    )
    coef_out = outputs_path("metrics", f"logreg_{short}_coefficients.csv")
    coef_df.to_csv(coef_out, index=False)

    val_acc = float((clf.predict(X_val_s) == y_val).mean()) if len(y_val) else 0.0
    val_auc = 0.0
    if len(y_val) and len(np.unique(y_val)) > 1:
        val_auc = float(roc_auc_score(y_val, clf.predict_proba(X_val_s)[:, 1]))
    metrics = {
        "target": args.target,
        "val_acc": val_acc,
        "val_auc": val_auc,
        "n_train": int((~is_val).sum()),
        "n_val": int(is_val.sum()),
        "note": "AUC is diagnostic only; gate is coverage@K",
    }
    metrics_path = outputs_path("metrics", f"logreg_{short}_train_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Saved %s | val_acc=%.3f val_auc=%.3f (diagnostic) | %s", ckpt, val_acc, val_auc, coef_out)


if __name__ == "__main__":
    main()
