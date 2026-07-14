#!/usr/bin/env python3
"""Train LightGBM LambdaRank evidence ranker.

Modes:
  --cv N         : image-level N-fold OOF scores (debug / leakage-safe)
  --final-train  : train on split=train images only (paper / VLM path)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import yaml

from src.features.ranker_features import FEATURE_KEYS, assert_no_feature_leakage
from src.utils.logging_utils import log_section, setup_experiment_logging
from src.utils.paths import outputs_path, repo_path


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
    raise FileNotFoundError("No ranker dataset. Run scripts/build_ranker_dataset.py first.")


def _lgbm_params(cfg: dict) -> dict:
    lgbm = cfg.get("ranker", {}).get("lgbm", {})
    return {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": lgbm.get("ndcg_eval_at", [1, 2, 4, 8]),
        "learning_rate": float(lgbm.get("learning_rate", 0.05)),
        "num_leaves": int(lgbm.get("num_leaves", 31)),
        "min_data_in_leaf": int(lgbm.get("min_data_in_leaf", 20)),
        "feature_fraction": float(lgbm.get("feature_fraction", 0.8)),
        "bagging_fraction": float(lgbm.get("bagging_fraction", 0.8)),
        "bagging_freq": int(lgbm.get("bagging_freq", 1)),
        "verbosity": -1,
    }


def _prepare_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """Sort by group_id and return group sizes for LightGBM."""
    df = df.sort_values(["group_id", "patch_index"]).reset_index(drop=True)
    sizes = df.groupby("group_id", sort=False).size().tolist()
    return df, sizes


def _fit_lambdarank(X, y, group, params, n_estimators: int):
    import lightgbm as lgb

    train_set = lgb.Dataset(X, label=y, group=group, feature_name=FEATURE_KEYS, free_raw_data=False)
    booster = lgb.train(params, train_set, num_boost_round=n_estimators)
    return booster


def _image_folds(image_ids: list[str], n_folds: int, seed: int) -> list[set[str]]:
    rng = np.random.RandomState(seed)
    ids = list(image_ids)
    rng.shuffle(ids)
    folds = [set() for _ in range(n_folds)]
    for i, iid in enumerate(ids):
        folds[i % n_folds].add(iid)
    return folds


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM LambdaRank ranker.")
    parser.add_argument("--objective", default="lambdarank")
    parser.add_argument("--target", choices=["strict_positive", "any_positive"], default="strict_positive")
    parser.add_argument("--cv", type=int, default=0, help="Image-level OOF folds (e.g. 5).")
    parser.add_argument("--final-train", action="store_true", help="Train on split=train only.")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--dataset-tag", default="100")
    parser.add_argument("--split-by", default="image_id")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logger = setup_experiment_logging("train_lgbm")
    log_section(logger, f"LGBM LambdaRank | target={args.target} cv={args.cv} final={args.final_train}")

    if args.objective != "lambdarank":
        raise SystemExit("Only --objective lambdarank is supported")
    if args.split_by != "image_id":
        raise SystemExit("--split-by must be image_id")
    if not args.cv and not args.final_train:
        # Default debug behavior: 5-fold OOF
        args.cv = 5
        logger.info("Neither --cv nor --final-train set; defaulting to --cv 5")

    try:
        import lightgbm as lgb  # noqa: F401
    except ImportError:
        logger.error("lightgbm not installed — pip install lightgbm")
        sys.exit(1)

    cfg = {}
    cfg_path = repo_path("configs", "qe_bops.yaml")
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    df = _load_dataset(args.dataset, args.dataset_tag)
    assert_no_feature_leakage(FEATURE_KEYS)
    missing = [k for k in FEATURE_KEYS if k not in df.columns]
    if missing:
        raise SystemExit(f"Dataset missing features: {missing}")

    # Relevance for LambdaRank must be non-negative ints
    df = df.copy()
    df["relevance"] = df[args.target].astype(int)
    params = _lgbm_params(cfg)
    n_estimators = int(cfg.get("ranker", {}).get("lgbm", {}).get("n_estimators", 300))
    short = "strict" if args.target == "strict_positive" else "any"
    ckpt_dir = outputs_path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    oof_dir = outputs_path("ranker")
    oof_dir.mkdir(parents=True, exist_ok=True)

    if args.cv:
        image_ids = sorted(df["image_id"].unique().tolist())
        folds = _image_folds(image_ids, args.cv, seed=int(cfg.get("manifests", {}).get("ranker_split_seed", 42)))
        oof_scores = pd.Series(np.nan, index=df.index, dtype=float)
        fold_models = []

        for fold_i, val_ids in enumerate(folds):
            train_mask = ~df["image_id"].isin(val_ids)
            val_mask = df["image_id"].isin(val_ids)
            train_df, train_group = _prepare_groups(df.loc[train_mask])
            val_df = df.loc[val_mask]

            X_train = train_df[FEATURE_KEYS].to_numpy(dtype=float)
            y_train = train_df["relevance"].to_numpy(dtype=int)
            booster = _fit_lambdarank(X_train, y_train, train_group, params, n_estimators)

            X_val = val_df[FEATURE_KEYS].to_numpy(dtype=float)
            preds = booster.predict(X_val)
            oof_scores.loc[val_df.index] = preds

            fold_path = ckpt_dir / f"lgbm_{short}_fold{fold_i}.txt"
            booster.save_model(str(fold_path))
            fold_models.append(str(fold_path))
            logger.info("Fold %d/%d | train_images=%d val_images=%d", fold_i + 1, args.cv,
                        train_df["image_id"].nunique(), val_df["image_id"].nunique())

        oof_df = df[["image_id", "question_id", "group_id", "patch_index", "q_bops_score",
                      "strict_positive", "any_positive", "x", "y", "w", "h"]].copy()
        oof_df[f"score_{short}"] = oof_scores.to_numpy()
        oof_path = oof_dir / f"oof_scores_{short}_{args.dataset_tag}.parquet"
        oof_df.to_parquet(oof_path, index=False)
        # Also write default path for eval
        oof_df.to_parquet(oof_dir / f"oof_scores_{short}.parquet", index=False)

        meta = {
            "target": args.target,
            "mode": "oof",
            "cv": args.cv,
            "features": FEATURE_KEYS,
            "fold_models": fold_models,
            "oof_path": str(oof_path),
            "dataset_tag": args.dataset_tag,
        }
        with open(ckpt_dir / f"lgbm_{short}_oof_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info("Wrote OOF scores %s", oof_path)

    if args.final_train:
        train_df, train_group = _prepare_groups(df.loc[df["split"] == "train"])
        if train_df.empty:
            raise SystemExit("No split=train rows for --final-train")
        X_train = train_df[FEATURE_KEYS].to_numpy(dtype=float)
        y_train = train_df["relevance"].to_numpy(dtype=int)
        booster = _fit_lambdarank(X_train, y_train, train_group, params, n_estimators)
        out = ckpt_dir / f"lgbm_{short}.txt"
        booster.save_model(str(out))
        with open(ckpt_dir / f"lgbm_{short}_features.json", "w", encoding="utf-8") as f:
            json.dump({"features": FEATURE_KEYS, "target": args.target, "mode": "final_train"}, f, indent=2)
        logger.info("Saved final model %s (train images=%d)", out, train_df["image_id"].nunique())


if __name__ == "__main__":
    main()
