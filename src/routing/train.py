"""Reusable OOF training/evaluation for the RAVEN-BOPS reader-aware router."""

from __future__ import annotations

from typing import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.metrics.statistical_tests import bootstrap_ci
from src.routing import build_router_features, router_feature_keys
from src.utils.paths import outputs_path

METHOD_FILES = {
    "resize": "vlm_metrics_{dataset}_{n}_resize_single{tag}.csv",
    "bm25": "vlm_metrics_{dataset}_{n}_bm25_only_k2{tag}.csv",
    "q_bops": "vlm_metrics_{dataset}_{n}_bops_qa_fair_pool_k2{tag}.csv",
    "ler_bops": "vlm_metrics_{dataset}_{n}_learned_lgbm_strict_k2{tag}.csv",
    "uniform": "vlm_metrics_{dataset}_{n}_uniform_k2{tag}.csv",
}


def load_methods(
    n: int,
    methods: list[str],
    *,
    metrics_tag: str = "",
    dataset: str = "docvqa",
) -> pd.DataFrame:
    tag = f"_{metrics_tag}" if metrics_tag else ""
    frames = {}
    for m in methods:
        path = outputs_path(
            "metrics",
            METHOD_FILES[m].format(dataset=dataset, n=n, tag=tag),
        )
        df = pd.read_csv(path)[
            ["image_id", "question", "ground_truth_answer", "parsed_prediction", "anls", "exact_match"]
        ].copy()
        frames[m] = df.drop_duplicates("image_id").set_index("image_id")
    ids = sorted(set.intersection(*[set(f.index) for f in frames.values()]))
    base = frames[methods[0]].loc[ids][["question", "ground_truth_answer"]].copy()
    for m in methods:
        base[f"anls__{m}"] = frames[m].loc[ids]["anls"].astype(float).values
        base[f"em__{m}"] = frames[m].loc[ids]["exact_match"].astype(float).values
        base[f"pred__{m}"] = frames[m].loc[ids]["parsed_prediction"].fillna("").astype(str).values
    base.index = ids
    return base


_REG_PARAMS = {
    "objective": "regression",
    "metric": "l2",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 10,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
}

_MARGIN_GRID = [0.0, 0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.30]


def _folds(n_items: int, n_splits: int = 5, seed: int = 42) -> list[np.ndarray]:
    rng = np.random.RandomState(seed)
    idx = np.arange(n_items)
    rng.shuffle(idx)
    return [idx[i::n_splits] for i in range(n_splits)]


def evaluate_router(
    n: int,
    methods: Sequence[str],
    *,
    default: str = "resize",
    seed: int = 42,
    drop_feature_prefixes: Sequence[str] | None = None,
    data: pd.DataFrame | None = None,
    metrics_tag: str = "",
) -> dict:
    """Train OOF per-method ANLS regressors and route with a default-biased margin.

    Args:
        drop_feature_prefixes: feature-name substrings to zero out (ablations),
            e.g. ["consensus", "agree"] to remove consensus features.
        metrics_tag: optional CSV suffix (e.g. ``qwen2vl2b``) for second-VLM runs.
    """
    methods = list(methods)
    if data is None:
        data = load_methods(n, methods, metrics_tag=metrics_tag)
    ids = list(data.index)
    default_j = methods.index(default)

    feat_keys = router_feature_keys(methods)
    drop = drop_feature_prefixes or []
    keep_mask = np.array([not any(d in k for d in drop) for k in feat_keys], dtype=bool)

    X = np.zeros((len(ids), len(feat_keys)), dtype=float)
    for i, iid in enumerate(ids):
        preds = {m: data.loc[iid, f"pred__{m}"] for m in methods}
        feats = build_router_features(iid, data.loc[iid, "question"], preds, methods, default_method=default)
        X[i] = [feats.get(k, 0.0) for k in feat_keys]
    X = X[:, keep_mask]

    anls_mat = np.column_stack([data[f"anls__{m}"].to_numpy() for m in methods])
    em_mat = np.column_stack([data[f"em__{m}"].to_numpy() for m in methods])
    oracle_anls = anls_mat.max(axis=1)

    single_anls = {m: float(data[f"anls__{m}"].mean()) for m in methods}
    single_em = {m: float(data[f"em__{m}"].mean()) for m in methods}
    best_base = max(single_anls, key=single_anls.get)

    def _route(pred: np.ndarray, margin: float) -> np.ndarray:
        alt = pred.argmax(axis=1)
        gain = pred[np.arange(len(pred)), alt] - pred[:, default_j]
        return np.where(gain > margin, alt, default_j)

    pred_anls = np.zeros_like(anls_mat)
    route_idx = np.zeros(len(ids), dtype=int)
    tuned_margins = []
    for val in _folds(len(ids), seed=seed):
        tr = np.ones(len(ids), dtype=bool)
        tr[val] = False
        pred_tr = np.zeros((int(tr.sum()), len(methods)))
        for j in range(len(methods)):
            booster = lgb.train(_REG_PARAMS, lgb.Dataset(X[tr], label=anls_mat[tr, j]), num_boost_round=300)
            pred_anls[val, j] = booster.predict(X[val])
            pred_tr[:, j] = booster.predict(X[tr])
        anls_tr = anls_mat[tr]
        best_m, best_s = 0.0, -1.0
        for mg in _MARGIN_GRID:
            s = anls_tr[np.arange(len(anls_tr)), _route(pred_tr, mg)].mean()
            if s > best_s:
                best_s, best_m = s, mg
        tuned_margins.append(best_m)
        route_idx[val] = _route(pred_anls[val], best_m)

    routed_anls = anls_mat[np.arange(len(ids)), route_idx]
    routed_em = em_mat[np.arange(len(ids)), route_idx]
    base_anls_vec = data[f"anls__{best_base}"].to_numpy()
    base_em_vec = data[f"em__{best_base}"].to_numpy()

    mean_d, lo, hi = bootstrap_ci((routed_anls - base_anls_vec).tolist())
    em_mean_d, em_lo, em_hi = bootstrap_ci((routed_em - base_em_vec).tolist())
    wins = int((routed_anls > base_anls_vec + 1e-9).sum())
    losses = int((routed_anls < base_anls_vec - 1e-9).sum())

    return {
        "n": len(ids),
        "methods": methods,
        "single_method_anls": single_anls,
        "single_method_em": single_em,
        "best_base": best_base,
        "best_base_anls": single_anls[best_base],
        "best_base_em": single_em[best_base],
        "oracle_anls": float(oracle_anls.mean()),
        "routed_anls": float(routed_anls.mean()),
        "routed_em": float(routed_em.mean()),
        "anls_delta": mean_d,
        "anls_ci95": [lo, hi],
        "anls_ci_lower_positive": bool(lo > 0),
        "em_delta": em_mean_d,
        "em_ci95": [em_lo, em_hi],
        "optimal_selection_rate": float((routed_anls >= oracle_anls - 1e-9).mean()),
        "default_only_optimal_rate": float((anls_mat[:, default_j] >= oracle_anls - 1e-9).mean()),
        "headroom_recovered_frac": float(
            (routed_anls.mean() - single_anls[best_base]) / max(oracle_anls.mean() - single_anls[best_base], 1e-9)
        ),
        "chosen_route_counts": {methods[j]: int((route_idx == j).sum()) for j in range(len(methods))},
        "tuned_margins": tuned_margins,
        "win_tie_loss": {"wins": wins, "ties": len(ids) - wins - losses, "losses": losses},
        "routed_anls_vec": routed_anls.tolist(),
        "route_idx": route_idx.tolist(),
    }


def evaluate_pre_router(
    n: int,
    methods: Sequence[str],
    *,
    default: str = "resize",
    seed: int = 42,
    manifest: str | None = None,
    include_candidates: bool = False,
    feature_cache: str | None = None,
    data=None,
) -> dict:
    """RAVEN-pre: route BEFORE VLM using retrieval/OCR features only.

    Evaluation uses cached per-path ANLS of the chosen route (simulates one VLM call).
    """
    from src.data.dataset_loader import iter_manifest
    from src.routing.pre_features import build_pre_router_features, pre_router_feature_keys
    from src.utils.paths import repo_path

    methods = list(methods)
    if data is None:
        data = load_methods(n, methods)
    ids = list(data.index)
    default_j = methods.index(default)

    man_path = repo_path(manifest or f"Data/manifests/docvqa_{n}.jsonl")
    man_rows = {r["image_id"]: r for r in iter_manifest(man_path)}

    feat_keys = pre_router_feature_keys()
    cache_path = outputs_path("metrics", feature_cache or f"raven_pre_features_n{n}.parquet")
    if cache_path.exists():
        feat_df = pd.read_parquet(cache_path)
        feat_df = feat_df.set_index("image_id")
        X = np.zeros((len(ids), len(feat_keys)), dtype=float)
        for i, iid in enumerate(ids):
            if iid in feat_df.index:
                X[i] = [float(feat_df.loc[iid].get(k, 0.0)) for k in feat_keys]
            else:
                X[i] = 0.0
    else:
        records = []
        X = np.zeros((len(ids), len(feat_keys)), dtype=float)
        for i, iid in enumerate(ids):
            row = man_rows.get(iid, {})
            feats = build_pre_router_features(
                iid,
                str(data.loc[iid, "question"]),
                image_path=row.get("image_path"),
                include_candidates=include_candidates,
            )
            X[i] = [float(feats.get(k, 0.0)) for k in feat_keys]
            records.append({"image_id": iid, **{k: float(feats.get(k, 0.0)) for k in feat_keys}})
        pd.DataFrame.from_records(records).to_parquet(cache_path, index=False)

    anls_mat = np.column_stack([data[f"anls__{m}"].to_numpy() for m in methods])
    em_mat = np.column_stack([data[f"em__{m}"].to_numpy() for m in methods])
    oracle_anls = anls_mat.max(axis=1)
    single_anls = {m: float(data[f"anls__{m}"].mean()) for m in methods}
    single_em = {m: float(data[f"em__{m}"].mean()) for m in methods}
    best_base = max(single_anls, key=single_anls.get)

    def _route(pred: np.ndarray, margin: float) -> np.ndarray:
        alt = pred.argmax(axis=1)
        gain = pred[np.arange(len(pred)), alt] - pred[:, default_j]
        return np.where(gain > margin, alt, default_j)

    pred_anls = np.zeros_like(anls_mat)
    route_idx = np.zeros(len(ids), dtype=int)
    tuned_margins = []
    for val in _folds(len(ids), seed=seed):
        tr = np.ones(len(ids), dtype=bool)
        tr[val] = False
        pred_tr = np.zeros((int(tr.sum()), len(methods)))
        for j in range(len(methods)):
            booster = lgb.train(_REG_PARAMS, lgb.Dataset(X[tr], label=anls_mat[tr, j]), num_boost_round=300)
            pred_anls[val, j] = booster.predict(X[val])
            pred_tr[:, j] = booster.predict(X[tr])
        anls_tr = anls_mat[tr]
        best_m, best_s = 0.0, -1.0
        for mg in _MARGIN_GRID:
            s = anls_tr[np.arange(len(anls_tr)), _route(pred_tr, mg)].mean()
            if s > best_s:
                best_s, best_m = s, mg
        tuned_margins.append(best_m)
        route_idx[val] = _route(pred_anls[val], best_m)

    routed_anls = anls_mat[np.arange(len(ids)), route_idx]
    routed_em = em_mat[np.arange(len(ids)), route_idx]
    base_anls_vec = data[f"anls__{best_base}"].to_numpy()
    base_em_vec = data[f"em__{best_base}"].to_numpy()
    mean_d, lo, hi = bootstrap_ci((routed_anls - base_anls_vec).tolist())
    em_mean_d, em_lo, em_hi = bootstrap_ci((routed_em - base_em_vec).tolist())
    wins = int((routed_anls > base_anls_vec + 1e-9).sum())
    losses = int((routed_anls < base_anls_vec - 1e-9).sum())

    return {
        "n": len(ids),
        "variant": "raven_pre",
        "vlm_calls": 1,
        "methods": methods,
        "feature_cache": str(cache_path),
        "include_candidates": include_candidates,
        "single_method_anls": single_anls,
        "single_method_em": single_em,
        "best_base": best_base,
        "best_base_anls": single_anls[best_base],
        "best_base_em": single_em[best_base],
        "oracle_anls": float(oracle_anls.mean()),
        "routed_anls": float(routed_anls.mean()),
        "routed_em": float(routed_em.mean()),
        "anls_delta": mean_d,
        "anls_ci95": [lo, hi],
        "anls_ci_lower_positive": bool(lo > 0),
        "em_delta": em_mean_d,
        "em_ci95": [em_lo, em_hi],
        "optimal_selection_rate": float((routed_anls >= oracle_anls - 1e-9).mean()),
        "default_only_optimal_rate": float((anls_mat[:, default_j] >= oracle_anls - 1e-9).mean()),
        "chosen_route_counts": {methods[j]: int((route_idx == j).sum()) for j in range(len(methods))},
        "tuned_margins": tuned_margins,
        "win_tie_loss": {"wins": wins, "ties": len(ids) - wins - losses, "losses": losses},
        "beats_best_single": bool(routed_anls.mean() > single_anls[best_base]),
    }
