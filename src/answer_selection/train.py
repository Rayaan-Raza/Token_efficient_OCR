"""OOF training for RAVEN-Select: score each route output, pick argmax per question."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.answer_selection.dataset import DEFAULT_METHODS, build_long_table, feature_matrix
from src.answer_selection.models import (
    fit_lgbm_lambdarank,
    fit_lgbm_regression,
    fit_logistic_best,
    fit_ridge,
    predict_model,
)
from src.metrics.statistical_tests import bootstrap_ci
from src.routing.train import load_methods


def _qid_folds(unique_ids: list[str], n_splits: int = 5, seed: int = 42) -> list[set[str]]:
    rng = np.random.RandomState(seed)
    ids = list(unique_ids)
    rng.shuffle(ids)
    folds = [set() for _ in range(n_splits)]
    for i, iid in enumerate(ids):
        folds[i % n_splits].add(iid)
    return folds


def _pick_argmax(scores: np.ndarray, routes: list[str], default: str = "resize") -> str:
    """Argmax with resize tie-break."""
    best_i = int(np.argmax(scores))
    best_s = float(scores[best_i])
    # Prefer default on ties
    for i, r in enumerate(routes):
        if abs(float(scores[i]) - best_s) < 1e-12 and r == default:
            return r
    return routes[best_i]


def oof_select(
    long_df: pd.DataFrame,
    *,
    model_name: str = "lgbm_reg",
    seed: int = 42,
    drop_groups: Sequence[str] | None = None,
    default: str = "resize",
    n_splits: int = 5,
) -> dict:
    """5-fold OOF by image_id; score each row; select argmax route per question."""
    X_df, keys = feature_matrix(long_df, drop_groups=drop_groups)
    X = X_df.to_numpy(dtype=float)
    y = long_df["label"].to_numpy(dtype=float)
    routes = long_df["route"].tolist()
    image_ids = long_df["image_id"].tolist()
    unique_ids = list(dict.fromkeys(image_ids))
    id_to_rows: dict[str, list[int]] = {}
    for i, iid in enumerate(image_ids):
        id_to_rows.setdefault(iid, []).append(i)

    # Binary best-route label for logistic
    y_best = np.zeros(len(long_df), dtype=int)
    for iid, idxs in id_to_rows.items():
        best_j = max(idxs, key=lambda j: y[j])
        y_best[best_j] = 1

    oof_scores = np.zeros(len(long_df), dtype=float)
    folds = _qid_folds(unique_ids, n_splits=n_splits, seed=seed)

    for val_ids in folds:
        val_mask = np.array([iid in val_ids for iid in image_ids], dtype=bool)
        tr_mask = ~val_mask
        X_tr, y_tr = X[tr_mask], y[tr_mask]
        X_va = X[val_mask]

        if model_name == "ridge":
            model = fit_ridge(X_tr, y_tr)
            oof_scores[val_mask] = predict_model(model, X_va)
        elif model_name == "logistic":
            model = fit_logistic_best(X_tr, y_best[tr_mask])
            oof_scores[val_mask] = predict_model(model, X_va)
        elif model_name == "lgbm_reg":
            model = fit_lgbm_regression(X_tr, y_tr)
            oof_scores[val_mask] = predict_model(model, X_va)
        elif model_name == "lgbm_rank":
            # Build group sizes in train order (contiguous by image_id)
            tr_idx = np.where(tr_mask)[0]
            # Reorder train so rows of same qid are contiguous
            order = sorted(tr_idx, key=lambda j: image_ids[j])
            X_tr_ord = X[order]
            y_tr_ord = y[order]
            groups: list[int] = []
            cur_id, cnt = None, 0
            for j in order:
                if image_ids[j] != cur_id:
                    if cnt:
                        groups.append(cnt)
                    cur_id, cnt = image_ids[j], 1
                else:
                    cnt += 1
            if cnt:
                groups.append(cnt)
            model = fit_lgbm_lambdarank(X_tr_ord, y_tr_ord, groups)
            oof_scores[val_mask] = predict_model(model, X_va)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    # Select per question
    chosen_route = []
    chosen_anls = []
    chosen_em = []
    route_counts: dict[str, int] = {}
    for iid in unique_ids:
        idxs = id_to_rows[iid]
        sc = np.array([oof_scores[j] for j in idxs])
        rts = [routes[j] for j in idxs]
        pick = _pick_argmax(sc, rts, default=default)
        # Find row for pick
        pick_j = next(j for j in idxs if routes[j] == pick)
        chosen_route.append(pick)
        chosen_anls.append(float(long_df.iloc[pick_j]["label"]))
        chosen_em.append(float(long_df.iloc[pick_j]["em"]))
        route_counts[pick] = route_counts.get(pick, 0) + 1

    return {
        "model": model_name,
        "feature_keys": keys,
        "n": len(unique_ids),
        "image_ids": unique_ids,
        "chosen_route": chosen_route,
        "anls_vec": chosen_anls,
        "em_vec": chosen_em,
        "anls": float(np.mean(chosen_anls)),
        "em": float(np.mean(chosen_em)),
        "route_counts": route_counts,
        "oof_scores": oof_scores.tolist(),
    }


def _pack_result(
    n: int,
    model_name: str,
    methods: list[str],
    anls_vec: list[float],
    em_vec: list[float],
    route_counts: dict,
    bases: dict,
    *,
    drop_groups: Sequence[str] | None = None,
    feature_keys: list[str] | None = None,
) -> dict:
    anls_arr = np.array(anls_vec, dtype=float)
    em_arr = np.array(em_vec, dtype=float)
    resize_anls = np.array(bases["resize"]["anls_vec"], dtype=float)
    short_anls = np.array(bases["shortest_nonempty"]["anls_vec"], dtype=float)
    d_resize = bootstrap_ci((anls_arr - resize_anls).tolist())
    d_short = bootstrap_ci((anls_arr - short_anls).tolist())
    from src.answer_selection.method_spec import METHOD_VERSION, method_stamp

    anls = float(anls_arr.mean())
    em = float(em_arr.mean())
    is_production = model_name in {
        "raven_select_rule",
        "rule",
        "raven_select",
        "ocr_present_shortest",
        "rule_ocr_present_shortest",
    }
    return {
        "n": n,
        "variant": "raven_select",
        "method": method_stamp(
            role="production" if is_production and not drop_groups else "comparator",
            comparator_model=model_name,
        ),
        "method_version": METHOD_VERSION,
        "vlm_calls": 3,
        "model": model_name,
        "methods": methods,
        "drop_groups": list(drop_groups or []),
        "anls": anls,
        "em": em,
        "route_counts": route_counts,
        "anls_vec": anls_vec,
        "em_vec": em_vec,
        "baselines": {
            k: {"anls": v["anls"], "em": v["em"], "route_counts": v["route_counts"]}
            for k, v in bases.items()
        },
        "vs_resize": {
            "delta": d_resize[0],
            "ci95": [d_resize[1], d_resize[2]],
            "ci_lower_positive": d_resize[1] > 0,
        },
        "vs_shortest_nonempty": {
            "delta": d_short[0],
            "ci95": [d_short[1], d_short[2]],
            "ci_lower_positive": d_short[1] > 0,
        },
        "beats_resize": bool(anls > bases["resize"]["anls"]),
        "beats_shortest_nonempty": bool(anls > bases["shortest_nonempty"]["anls"]),
        "target_anls_met": bool(anls >= 0.805),
        "target_em_met": bool(em >= 0.705),
        "feature_keys": feature_keys or [],
        "p14_pass": bool(anls > bases["shortest_nonempty"]["anls"]),
        "p15_pass": bool(d_resize[1] > 0 and d_short[1] > 0),
    }


def evaluate_selector(
    n: int,
    *,
    methods: Sequence[str] | None = None,
    model_name: str = "lgbm_reg",
    seed: int = 42,
    drop_groups: Sequence[str] | None = None,
    metrics_tag: str = "",
    default: str = "resize",
    long_df: pd.DataFrame | None = None,
) -> dict:
    """Full RAVEN-Select evaluation with comparisons vs resize and shortest_nonempty."""
    from src.answer_selection.baselines import evaluate_baselines
    from src.answer_selection.dataset import _load_ocr_presence

    methods = list(methods or DEFAULT_METHODS)
    data = load_methods(n, methods, metrics_tag=metrics_tag)
    ocr = _load_ocr_presence(n, metrics_tag=metrics_tag)
    bases = evaluate_baselines(data, methods, ocr_presence=ocr, default=default)

    if model_name in ("ocr_present_shortest", "rule_ocr_present_shortest", "raven_select_rule", "rule", "raven_select"):
        from src.answer_selection.baselines import evaluate_baselines as _eb

        from src.answer_selection.method_spec import PRODUCTION_FLAGS

        # Frozen production flags; ablations may toggle groups via drop_groups.
        flags = dict(PRODUCTION_FLAGS)
        if drop_groups:
            if "ocr_presence" in drop_groups:
                flags["use_ocr"] = False
            if "answer_type" in drop_groups:
                flags["use_answer_type"] = True  # force-on shows type is not needed / can hurt
            if "pred_text" in drop_groups:
                flags["use_length"] = False
            if "consensus" in drop_groups:
                flags["use_consensus"] = True  # force-on
        ruled = _eb(data, methods, ocr_presence=ocr, default=default, rule_flags=flags)
        b = ruled["raven_select_rule"]
        return _pack_result(
            n, "raven_select_rule", methods,
            b["anls_vec"], b["em_vec"], b["route_counts"], bases,
            drop_groups=drop_groups,
        )

    if long_df is None:
        long_df = build_long_table(n, methods, metrics_tag=metrics_tag, data=data)

    sel = oof_select(
        long_df,
        model_name=model_name,
        seed=seed,
        drop_groups=drop_groups,
        default=default,
    )
    return _pack_result(
        n, model_name, methods,
        sel["anls_vec"], sel["em_vec"], sel["route_counts"], bases,
        drop_groups=drop_groups,
        feature_keys=sel["feature_keys"],
    )
