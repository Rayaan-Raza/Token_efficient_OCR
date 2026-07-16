"""Build long (qid × route) tables for RAVEN-Select from cached VLM outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from src.answer_selection.features import build_output_features, feature_keys
from src.routing.train import load_methods
from src.utils.paths import outputs_path

DEFAULT_METHODS = ["resize", "bm25", "ler_bops"]

_SCORE_COLS = [
    "bm25_top1", "bm25_top2", "bm25_gap", "bm25_mean", "bm25_std",
    "ler_top1", "ler_top2", "ler_gap", "ler_mean", "ler_std",
    "retrieval_jaccard", "retrieval_overlap_count",
    "ocr_n_boxes", "ocr_mean_conf", "ocr_total_chars", "ocr_total_tokens",
]


def _load_pre_scores(n: int) -> pd.DataFrame | None:
    path = outputs_path("metrics", f"raven_pre_features_n{n}.parquet")
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "image_id" not in df.columns:
        return None
    return df.set_index("image_id")


def _load_ocr_presence(
    n: int,
    path: Path | None = None,
    *,
    metrics_tag: str = "",
) -> pd.DataFrame | None:
    suffix = f"_{metrics_tag}" if metrics_tag else ""
    p = path or outputs_path("metrics", f"raven_select_ocr_presence_n{n}{suffix}.parquet")
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    return df.set_index(["image_id", "route"])


def build_long_table(
    n: int,
    methods: Sequence[str] | None = None,
    *,
    metrics_tag: str = "",
    ocr_presence_path: Path | None = None,
    data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Expand wide method frame into one row per (image_id, route).

    Columns include features, label (ANLS), em, prediction, question, gold
    (gold only for labels/metrics — never passed into model feature matrix).
    """
    methods = list(methods or DEFAULT_METHODS)
    if data is None:
        data = load_methods(n, methods, metrics_tag=metrics_tag)
    pre = _load_pre_scores(n)
    ocr = _load_ocr_presence(n, ocr_presence_path, metrics_tag=metrics_tag)
    keys = feature_keys(methods)

    rows: list[dict] = []
    for iid in data.index:
        question = str(data.loc[iid, "question"])
        gold = str(data.loc[iid, "ground_truth_answer"])
        preds = {m: str(data.loc[iid, f"pred__{m}"]) for m in methods}
        scores = {}
        if pre is not None and iid in pre.index:
            scores = {c: float(pre.loc[iid].get(c, 0.0)) for c in _SCORE_COLS if c in pre.columns}

        for m in methods:
            full_ocr = ""
            patch_ocr = ""
            if ocr is not None and (iid, m) in ocr.index:
                r = ocr.loc[(iid, m)]
                full_ocr = str(r.get("full_ocr_text", "") or "")
                patch_ocr = str(r.get("patch_ocr_text", "") or "")
            feats = build_output_features(
                m,
                preds[m],
                question,
                preds,
                methods=methods,
                ocr_full_text=full_ocr,
                ocr_patch_text=patch_ocr,
                route_scores=scores,
            )
            rows.append({
                "image_id": iid,
                "route": m,
                "question": question,
                "prediction": preds[m],
                "gold": gold,
                "label": float(data.loc[iid, f"anls__{m}"]),
                "em": float(data.loc[iid, f"em__{m}"]),
                **{k: float(feats.get(k, 0.0)) for k in keys},
            })
    return pd.DataFrame(rows)


def feature_matrix(
    long_df: pd.DataFrame,
    *,
    drop_groups: Sequence[str] | None = None,
    keys: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Return (X dataframe, kept feature key list), optionally dropping groups."""
    from src.answer_selection.features import FEATURE_GROUPS

    keys = list(keys or feature_keys())
    drop = drop_groups or []
    if drop:
        drop_prefs: list[str] = []
        for g in drop:
            drop_prefs.extend(FEATURE_GROUPS.get(g, [g]))

        def _keep(k: str) -> bool:
            return not any(p in k or k.startswith(p) or k == p for p in drop_prefs)

        keys = [k for k in keys if _keep(k)]
    X = long_df[keys].astype(float).copy()
    return X, keys
