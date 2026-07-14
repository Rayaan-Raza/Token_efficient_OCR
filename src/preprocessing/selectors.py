"""Fair patch selector methods for QE-BOPS evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from PIL import Image

from src.metrics.answer_coverage import oracle_select_patches
from src.preprocessing.bops import select_ocr_guided_patches, select_random_patches, select_uniform_patches
from src.preprocessing.candidate_pool import (
    CandidatePoolConfig,
    generate_candidate_pool,
    generate_original_grid,
    pool_hash,
)
from src.preprocessing.mmr_select import mmr_select, patch_iou
from src.preprocessing.patch_grid import Patch, crop_patch
from src.preprocessing.patch_scoring import score_patch, text_confidence_score
from src.preprocessing.patch_scoring_qa import score_patch_question_aware
from src.preprocessing.qe_bops_entity_row import select_entity_row_patches
from src.preprocessing.qe_bops_node_pair import select_node_pair_patches
from src.preprocessing.qe_bops_table_pair import select_table_pair_patches
from src.preprocessing.qe_bops_scoring import (
    retrieve_twostage_pool,
    score_qe_bops_patch,
)
from src.features.patch_features import bm25_score, answer_type_score, extract_patch_features


class AnswerLeakageError(ValueError):
    """Raised when GT answers are passed to inference-time selection."""


FAIR_METHODS = {
    "random",
    "uniform",
    "bops_fair_pool",
    "bops_qa_fair_pool",
    "ocr_confidence_topk",
    "bm25_only",
    "question_overlap_topk",
    "answer_type_only",
    "multiscale_uniform",
    "edge_strip",
    "qe_bops",
    "qe_bops_v1",
    "qe_bops_no_question",
    "qe_bops_v2",
    "qe_bops_topk",
    "qe_bops_twostage",
    "qe_bops_anchor_pair",
    "qe_bops_anchor_evidence",
    "qe_bops_pair_top8",
    "qe_bops_safe_expand",
    "qe_bops_node_pair",
    "qe_bops_table_pair",
    "qe_bops_entity_row",
    "learned_logreg",
    "learned_lgbm_strict",
    "learned_lgbm_any",
    "learned_lgbm_combined",
    "learned_lgbm_qbops_hybrid",
}

DIAGNOSTIC_METHODS = {"oracle"}


@dataclass
class SelectionResult:
    method: str
    patches: list[Patch]
    patch_images: list[Image.Image]
    scores: list[float]
    candidate_pool: list[Patch]
    pool_hash_value: str
    meta: dict[str, Any]


def _reject_answers(method: str, answers: list[str] | None, eval_labels: bool) -> None:
    if answers and method not in DIAGNOSTIC_METHODS and not eval_labels:
        raise AnswerLeakageError(
            f"Ground-truth answers cannot be used for method={method} at inference time."
        )


def _build_fair_pool(
    image: Image.Image,
    question: str,
    ocr_boxes: list[dict[str, Any]],
    *,
    include_question_centered: bool = True,
) -> list[Patch]:
    cfg = CandidatePoolConfig(include_question_token_centered=include_question_centered)
    return generate_candidate_pool(
        image, question, ocr_boxes, cfg, include_question_centered=include_question_centered
    )


def _select_qe_bops(
    image: Image.Image,
    pool: list[Patch],
    k: int,
    question: str,
    ocr_boxes: list[dict[str, Any]],
    *,
    include_question: bool = True,
    version: str = "v1",
    mmr_lambda: float = 0.75,
    use_mmr: bool = True,
) -> tuple[list[Patch], list[float], list[float], dict[str, float]]:
    all_scores: list[float] = []
    feat_map: dict[str, float] = {}
    best_s = float("-inf")
    for p in pool:
        s, feats = score_qe_bops_patch(
            image, p, ocr_boxes, question,
            include_question=include_question, version=version,
        )
        all_scores.append(s)
        if s >= best_s:
            best_s = s
            feat_map = feats
    if use_mmr:
        selected = mmr_select(pool, all_scores, k, lambda_=mmr_lambda)
    else:
        ranked = sorted(zip(pool, all_scores), key=lambda x: x[1], reverse=True)
        selected = [p for p, _ in ranked[:k]]
    scores = [all_scores[pool.index(p)] for p in selected]
    return selected, scores, all_scores, feat_map


def _evidence_proxy_score(feats: dict[str, float]) -> float:
    """Lexical + label-value score for second-slot selection."""
    return (
        0.30 * feats.get("question_overlap", 0.0)
        + 0.30 * feats.get("bm25", 0.0)
        + 0.15 * feats.get("label_value_proximity", 0.0)
        + 0.10 * feats.get("same_row_label_value", 0.0)
        + 0.05 * feats.get("below_label_relation", 0.0)
        + 0.10 * feats.get("answer_type", 0.0)
    )


def _select_qe_bops_anchor_pair(
    image: Image.Image,
    pool: list[Patch],
    k: int,
    question: str,
    ocr_boxes: list[dict[str, Any]],
    *,
    redundancy_iou: float = 0.35,
    version: str = "v2",
    top_k_pool: int = 8,
    second_slot: str = "qe_topk",
) -> tuple[list[Patch], list[float], list[float], dict[str, float]]:
    """Q-BOPS-anchored pair: slot-1 = top Q-BOPS, slot-2+ = best QE evidence, non-redundant."""
    qa_scores = [score_patch_question_aware(image, p, ocr_boxes, question) for p in pool]
    first = max(zip(pool, qa_scores), key=lambda x: (x[1], -x[0].index))[0]

    all_scores: list[float] = []
    feat_by_index: dict[int, dict[str, float]] = {}
    feat_map: dict[str, float] = {}
    best_s = float("-inf")
    for p in pool:
        s, feats = score_qe_bops_patch(
            image, p, ocr_boxes, question, version=version,
        )
        all_scores.append(s)
        feat_by_index[p.index] = feats
        if s >= best_s:
            best_s = s
            feat_map = feats

    ranked_qe = sorted(
        zip(pool, all_scores),
        key=lambda x: (x[1], -x[0].index),
        reverse=True,
    )
    candidates = [p for p, _ in ranked_qe[:top_k_pool]] if second_slot == "evidence_topk" else [p for p, _ in ranked_qe]

    selected = [first]
    for p in candidates:
        if p.index == first.index:
            continue
        if patch_iou(p, first) >= redundancy_iou:
            continue
        selected.append(p)
        if len(selected) >= k:
            break

    if second_slot == "evidence_topk" and len(selected) < k:
        ranked_ev = sorted(
            pool,
            key=lambda p: (_evidence_proxy_score(feat_by_index[p.index]), -p.index),
            reverse=True,
        )
        for p in ranked_ev[:top_k_pool]:
            if p.index in {s.index for s in selected}:
                continue
            if patch_iou(p, first) >= redundancy_iou:
                continue
            selected.append(p)
            if len(selected) >= k:
                break

    if len(selected) < k:
        for p, _ in ranked_qe:
            if p.index not in {s.index for s in selected}:
                selected.append(p)
            if len(selected) >= k:
                break

    scores = []
    for p in selected:
        if p.index == first.index:
            scores.append(qa_scores[pool.index(p)])
        elif second_slot == "evidence_topk":
            scores.append(_evidence_proxy_score(feat_by_index[p.index]))
        else:
            scores.append(all_scores[pool.index(p)])
    return selected, scores, all_scores, feat_map


def _pair_selection_score(
    p1: Patch,
    p2: Patch,
    feats1: dict[str, float],
    feats2: dict[str, float],
    qa1: float,
    qa2: float,
) -> float:
    rel = max(
        qa1, qa2,
        feats1.get("question_overlap", 0.0),
        feats2.get("question_overlap", 0.0),
        feats1.get("bm25", 0.0),
        feats2.get("bm25", 0.0),
    )
    ev = _evidence_proxy_score(feats1) + _evidence_proxy_score(feats2)
    div = 1.0 - patch_iou(p1, p2)
    return 0.40 * rel + 0.50 * ev + 0.10 * div


def _select_qe_bops_pair_top8(
    image: Image.Image,
    pool: list[Patch],
    k: int,
    question: str,
    ocr_boxes: list[dict[str, Any]],
    *,
    redundancy_iou: float = 0.35,
    version: str = "v2",
    top_qa: int = 3,
    top_qe: int = 8,
) -> tuple[list[Patch], list[float], list[float], dict[str, float]]:
    """Score patch pairs from Q-BOPS top-3 x QE top-8 for K=2 precision."""
    qa_scores = [score_patch_question_aware(image, p, ocr_boxes, question) for p in pool]
    ranked_qa = sorted(zip(pool, qa_scores), key=lambda x: (x[1], -x[0].index), reverse=True)
    qa_shortlist = [p for p, _ in ranked_qa[:top_qa]]

    all_scores: list[float] = []
    feat_by_index: dict[int, dict[str, float]] = {}
    feat_map: dict[str, float] = {}
    best_s = float("-inf")
    for p in pool:
        s, feats = score_qe_bops_patch(
            image, p, ocr_boxes, question, version=version,
        )
        all_scores.append(s)
        feat_by_index[p.index] = feats
        if s >= best_s:
            best_s = s
            feat_map = feats

    ranked_qe = sorted(zip(pool, all_scores), key=lambda x: (x[1], -x[0].index), reverse=True)
    qe_shortlist = [p for p, _ in ranked_qe[:top_qe]]

    best_pair: tuple[Patch, Patch] | None = None
    best_pair_score = float("-inf")
    for p1 in qa_shortlist:
        f1 = feat_by_index[p1.index]
        q1 = qa_scores[pool.index(p1)]
        for p2 in qe_shortlist:
            if p1.index == p2.index:
                continue
            if patch_iou(p1, p2) >= redundancy_iou:
                continue
            f2 = feat_by_index[p2.index]
            q2 = qa_scores[pool.index(p2)]
            ps = _pair_selection_score(p1, p2, f1, f2, q1, q2)
            if ps > best_pair_score:
                best_pair_score = ps
                best_pair = (p1, p2)

    if best_pair is None:
        return _select_qe_bops_anchor_pair(
            image, pool, k, question, ocr_boxes, version=version, second_slot="evidence_topk",
        )

    selected = list(best_pair)[:k]
    scores = [
        qa_scores[pool.index(selected[0])],
        _evidence_proxy_score(feat_by_index[selected[1].index]),
    ]
    return selected, scores, all_scores, feat_map


def _select_qe_bops_safe_expand(
    image: Image.Image,
    pool: list[Patch],
    k: int,
    question: str,
    ocr_boxes: list[dict[str, Any]],
    *,
    redundancy_iou: float = 0.35,
    version: str = "v2",
    top_qe: int = 8,
    swap_margin: float = 0.08,
) -> tuple[list[Patch], list[float], list[float], dict[str, float]]:
    """Start from Q-BOPS MMR@K; swap slot-2 only if QE top-8 evidence is clearly better."""
    qa_scores = [score_patch_question_aware(image, p, ocr_boxes, question) for p in pool]
    qa_sel = mmr_select(pool, qa_scores, k, lambda_=0.5)
    if len(qa_sel) < 2:
        return qa_sel, [qa_scores[pool.index(p)] for p in qa_sel], qa_scores, {}

    all_scores: list[float] = []
    feat_by_index: dict[int, dict[str, float]] = {}
    feat_map: dict[str, float] = {}
    best_s = float("-inf")
    for p in pool:
        s, feats = score_qe_bops_patch(
            image, p, ocr_boxes, question, version=version,
        )
        all_scores.append(s)
        feat_by_index[p.index] = feats
        if s >= best_s:
            best_s = s
            feat_map = feats

    p1, p2_q = qa_sel[0], qa_sel[1]
    q2_proxy = _evidence_proxy_score(feat_by_index[p2_q.index])
    ranked_qe = sorted(zip(pool, all_scores), key=lambda x: (x[1], -x[0].index), reverse=True)

    best_ev: Patch | None = None
    best_ev_proxy = q2_proxy
    for p, _ in ranked_qe[:top_qe]:
        if p.index in {p1.index, p2_q.index}:
            continue
        if patch_iou(p, p1) >= redundancy_iou:
            continue
        proxy = _evidence_proxy_score(feat_by_index[p.index])
        if proxy > best_ev_proxy:
            best_ev_proxy = proxy
            best_ev = p

    selected = [p1, p2_q]
    if best_ev is not None and best_ev_proxy >= q2_proxy + swap_margin:
        selected[1] = best_ev

    scores = [qa_scores[pool.index(selected[0])]]
    scores.append(
        _evidence_proxy_score(feat_by_index[selected[1].index])
        if selected[1].index != p2_q.index
        else qa_scores[pool.index(selected[1])]
    )
    return selected, scores, all_scores, feat_map


def _load_oof_score_map(
    image_id: str,
    *,
    score_col: str = "score_strict",
    question_id: str | None = None,
) -> dict[tuple[int, int, int, int], float] | None:
    """Load OOF patch scores for one image (leakage-safe). Prefer tag 500 then 100."""
    import pandas as pd

    from src.utils.paths import outputs_path

    candidates = [
        outputs_path("ranker", "oof_scores_strict_500.parquet"),
        outputs_path("ranker", "oof_scores_any_500.parquet"),
        outputs_path("ranker", "oof_scores_strict_100.parquet"),
        outputs_path("ranker", "oof_scores_strict.parquet"),
    ]
    # Prefer score-column-matching files
    if score_col == "score_any":
        candidates = [
            outputs_path("ranker", "oof_scores_any_500.parquet"),
            outputs_path("ranker", "oof_scores_any_100.parquet"),
            outputs_path("ranker", "oof_scores_any.parquet"),
        ] + candidates

    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return None
    df = pd.read_parquet(path)
    sub = df[df["image_id"] == image_id]
    if sub.empty:
        return None
    if question_id and "question_id" in sub.columns:
        qsub = sub[sub["question_id"].astype(str) == str(question_id)]
        if not qsub.empty:
            sub = qsub
    col = score_col if score_col in sub.columns else (
        "score_strict" if "score_strict" in sub.columns else None
    )
    if col is None:
        return None
    out: dict[tuple[int, int, int, int], float] = {}
    for r in sub.itertuples():
        key = (int(r.x), int(r.y), int(r.w), int(r.h))
        out[key] = float(getattr(r, col))
    return out


def _select_learned_ranker(
    image: Image.Image,
    candidates: list[Patch],
    k: int,
    question: str,
    ocr_boxes: list[dict[str, Any]],
    method: str,
    *,
    image_id: str | None = None,
    question_id: str | None = None,
    prefer_oof: bool = True,
) -> tuple[list[Patch], list[float], list[float], dict[str, Any]]:
    """Score fair-pool candidates with OOF scores (preferred) or checkpoint; plain top-K."""
    import joblib
    import numpy as np
    import yaml

    from src.features.ranker_features import FEATURE_KEYS, extract_ranker_features_for_patches
    from src.utils.paths import outputs_path, repo_path

    cfg = {}
    cfg_path = repo_path("configs", "qe_bops.yaml")
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    comb = cfg.get("ranker", {}).get("combined", {})
    hyb = cfg.get("ranker", {}).get("qbops_hybrid", {})
    w_s = float(comb.get("strict_weight", 0.6))
    w_a = float(comb.get("any_weight", 0.4))
    w_l = float(hyb.get("learned_weight", 0.5))
    w_q = float(hyb.get("qbops_weight", 0.5))

    meta: dict[str, Any] = {"selection": "topk", "learned_method": method}

    # --- Prefer leakage-safe OOF scores when available ---
    if prefer_oof and image_id and method in (
        "learned_lgbm_strict", "learned_lgbm_any", "learned_lgbm_combined", "learned_lgbm_qbops_hybrid",
    ):
        score_maps: dict[str, dict[tuple[int, int, int, int], float]] = {}
        if method in ("learned_lgbm_strict", "learned_lgbm_combined", "learned_lgbm_qbops_hybrid"):
            m = _load_oof_score_map(image_id, score_col="score_strict", question_id=question_id)
            if m:
                score_maps["strict"] = m
        if method in ("learned_lgbm_any", "learned_lgbm_combined", "learned_lgbm_qbops_hybrid"):
            m = _load_oof_score_map(image_id, score_col="score_any", question_id=question_id)
            if m:
                score_maps["any"] = m
            elif "strict" in score_maps and method == "learned_lgbm_any":
                score_maps["any"] = score_maps["strict"]

        if score_maps:
            all_scores: list[float] = []
            for p in candidates:
                key = (int(p.x), int(p.y), int(p.w), int(p.h))
                s_strict = score_maps.get("strict", {}).get(key)
                s_any = score_maps.get("any", {}).get(key)
                if method == "learned_lgbm_strict":
                    score = float(s_strict) if s_strict is not None else float("-inf")
                elif method == "learned_lgbm_any":
                    score = float(s_any) if s_any is not None else float("-inf")
                elif method == "learned_lgbm_combined":
                    if s_strict is None and s_any is None:
                        score = float("-inf")
                    else:
                        score = w_s * float(s_strict or 0.0) + w_a * float(s_any or 0.0)
                else:  # hybrid: blend with q_bops from features only if needed
                    # Fall through to feature path for proper q_bops hybrid if missing OOF any
                    score = float("-inf")
                    if s_strict is not None or s_any is not None:
                        learned = w_s * float(s_strict or 0.0) + w_a * float(s_any or s_strict or 0.0)
                        score = learned  # temporary; hybrid re-blends below if we have q_bops
                all_scores.append(score)

            if method == "learned_lgbm_qbops_hybrid" and any(np.isfinite(all_scores)):
                # Need q_bops for hybrid blend — extract features once
                feat_rows = extract_ranker_features_for_patches(image, candidates, ocr_boxes, question)
                q_scores = np.array([r["q_bops_score"] for r in feat_rows], dtype=float)
                learned = np.array([
                    (w_s * float(score_maps.get("strict", {}).get((p.x, p.y, p.w, p.h), 0.0))
                     + w_a * float(score_maps.get("any", {}).get(
                         (p.x, p.y, p.w, p.h),
                         score_maps.get("strict", {}).get((p.x, p.y, p.w, p.h), 0.0),
                     )))
                    for p in candidates
                ], dtype=float)

                def _norm(v: np.ndarray) -> np.ndarray:
                    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
                    if not np.isfinite(lo) or hi <= lo:
                        return np.zeros_like(v)
                    return (v - lo) / (hi - lo)

                all_scores = (w_l * _norm(learned) + w_q * _norm(q_scores)).tolist()

            if any(np.isfinite(all_scores)) and max(all_scores) > float("-inf"):
                ranked = sorted(zip(candidates, all_scores), key=lambda x: x[1], reverse=True)
                selected = [p for p, _ in ranked[:k]]
                scores = [sc for _, sc in ranked[:k]]
                meta["score_source"] = "oof"
                return selected, scores, all_scores, meta

    # --- Checkpoint fallback (not OOF-safe if image was in train) ---
    feat_rows = extract_ranker_features_for_patches(image, candidates, ocr_boxes, question)
    X = np.array([[r[f] for f in FEATURE_KEYS] for r in feat_rows], dtype=float)
    q_scores = np.array([r["q_bops_score"] for r in feat_rows], dtype=float)

    def _predict_lgbm(short: str) -> np.ndarray:
        import lightgbm as lgb
        path = outputs_path("checkpoints", f"lgbm_{short}.txt")
        if not path.exists():
            alt = outputs_path("checkpoints", f"lgbm_{short}_fold0.txt")
            path = alt if alt.exists() else path
        if not path.exists():
            raise FileNotFoundError(f"Missing LGBM checkpoint for {short}")
        return np.asarray(lgb.Booster(model_file=str(path)).predict(X), dtype=float)

    def _predict_logreg() -> np.ndarray:
        path = outputs_path("checkpoints", "logreg_strict.pkl")
        if not path.exists():
            path = outputs_path("checkpoints", "logreg_any.pkl")
        blob = joblib.load(path)
        Xs = blob["scaler"].transform(X)
        return blob["model"].predict_proba(Xs)[:, 1]

    if method == "learned_logreg":
        all_scores = _predict_logreg().tolist()
    elif method == "learned_lgbm_strict":
        all_scores = _predict_lgbm("strict").tolist()
    elif method == "learned_lgbm_any":
        all_scores = _predict_lgbm("any").tolist()
    elif method == "learned_lgbm_combined":
        s = _predict_lgbm("strict")
        a = _predict_lgbm("any")
        all_scores = (w_s * s + w_a * a).tolist()
    elif method == "learned_lgbm_qbops_hybrid":
        s = _predict_lgbm("strict")
        a = _predict_lgbm("any")
        learned = w_s * s + w_a * a

        def _norm(v: np.ndarray) -> np.ndarray:
            lo, hi = float(v.min()), float(v.max())
            if hi <= lo:
                return np.zeros_like(v)
            return (v - lo) / (hi - lo)

        all_scores = (w_l * _norm(learned) + w_q * _norm(q_scores)).tolist()
    else:
        raise ValueError(method)

    ranked = sorted(zip(candidates, all_scores), key=lambda x: x[1], reverse=True)
    selected = [p for p, _ in ranked[:k]]
    scores = [sc for _, sc in ranked[:k]]
    meta["score_source"] = "checkpoint"
    return selected, scores, all_scores, meta


def select_patches(
    image: Image.Image,
    method: str,
    k: int,
    question: str,
    ocr_boxes: list[dict[str, Any]],
    *,
    seed: int = 0,
    answers: list[str] | None = None,
    patch_labels: list[dict[str, Any]] | None = None,
    eval_labels: bool = False,
    mmr_lambda: float = 0.75,
    image_id: str | None = None,
    question_id: str | None = None,
    prefer_oof: bool = True,
) -> SelectionResult:
    """Select K patches under fair-comparison protocol."""
    _reject_answers(method, answers, eval_labels)

    if method == "bops_original":
        candidates = generate_original_grid(image)
    elif method == "qe_bops_no_question":
        candidates = _build_fair_pool(image, question, ocr_boxes, include_question_centered=False)
    else:
        candidates = _build_fair_pool(image, question, ocr_boxes, include_question_centered=True)

    qid = question[:32]
    ph = pool_hash(candidates, qid)

    scores: list[float] = []
    selected: list[Patch] = []
    all_scores: list[float] = []
    top_feats: dict[str, float] = {}

    if method == "random":
        selected = select_random_patches(candidates, k, seed=seed)
        all_scores = [float(i) for i in range(len(candidates))]
    elif method == "uniform":
        selected = select_uniform_patches(candidates, k)
        all_scores = [float(i) for i in range(len(candidates))]
    elif method in ("bops_fair_pool", "bops_original"):
        selected, scores = select_ocr_guided_patches(image, candidates, k, ocr_boxes)
        all_scores = [score_patch(image, p, ocr_boxes) for p in candidates]
    elif method == "bops_qa_fair_pool":
        for p in candidates:
            scores.append(score_patch_question_aware(image, p, ocr_boxes, question))
        all_scores = list(scores)
        selected = mmr_select(candidates, scores, k, lambda_=0.5)
        scores = [scores[candidates.index(p)] for p in selected]
    elif method == "ocr_confidence_topk":
        all_scores = [text_confidence_score(p, ocr_boxes) for p in candidates]
        scores = list(all_scores)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        selected = [p for p, _ in ranked[:k]]
    elif method == "bm25_only":
        from src.preprocessing.patch_scoring_qa import patch_ocr_text

        all_scores = [bm25_score(question, patch_ocr_text(p, ocr_boxes)) for p in candidates]
        scores = list(all_scores)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        selected = [p for p, _ in ranked[:k]]
    elif method == "question_overlap_topk":
        from src.preprocessing.patch_scoring_qa import question_token_overlap_score

        all_scores = [question_token_overlap_score(p, ocr_boxes, question) for p in candidates]
        scores = list(all_scores)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        selected = [p for p, _ in ranked[:k]]
    elif method == "answer_type_only":
        from src.preprocessing.patch_scoring_qa import patch_ocr_text

        all_scores = [answer_type_score(question, patch_ocr_text(p, ocr_boxes)) for p in candidates]
        scores = list(all_scores)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        selected = [p for p, _ in ranked[:k]]
    elif method == "multiscale_uniform":
        selected = select_uniform_patches(candidates, k)
        all_scores = [float(i) for i in range(len(candidates))]
    elif method == "edge_strip":
        edge = [p for p in candidates if p.y <= 5 or p.x <= 5][:k]
        selected = edge if len(edge) >= k else select_uniform_patches(candidates, k)
        all_scores = [float(i) for i in range(len(candidates))]
    elif method == "qe_bops":
        selected, scores, all_scores, top_feats = _select_qe_bops(
            image, candidates, k, question, ocr_boxes,
            version="v2", mmr_lambda=mmr_lambda, use_mmr=True,
        )
    elif method == "qe_bops_v1":
        selected, scores, all_scores, top_feats = _select_qe_bops(
            image, candidates, k, question, ocr_boxes,
            version="v1", mmr_lambda=mmr_lambda, use_mmr=True,
        )
    elif method == "qe_bops_v2":
        selected, scores, all_scores, top_feats = _select_qe_bops(
            image, candidates, k, question, ocr_boxes,
            version="v2", mmr_lambda=mmr_lambda, use_mmr=True,
        )
    elif method == "qe_bops_topk":
        selected, scores, all_scores, top_feats = _select_qe_bops(
            image, candidates, k, question, ocr_boxes,
            version="v1", mmr_lambda=mmr_lambda, use_mmr=False,
        )
    elif method == "qe_bops_twostage":
        pool = retrieve_twostage_pool(image, candidates, ocr_boxes, question)
        selected, scores, all_scores, top_feats = _select_qe_bops(
            image, pool, k, question, ocr_boxes,
            version="v2", mmr_lambda=mmr_lambda, use_mmr=True,
        )
        candidates = pool
    elif method == "qe_bops_anchor_pair":
        selected, scores, all_scores, top_feats = _select_qe_bops_anchor_pair(
            image, candidates, k, question, ocr_boxes, version="v2", second_slot="qe_topk",
        )
    elif method == "qe_bops_anchor_evidence":
        selected, scores, all_scores, top_feats = _select_qe_bops_anchor_pair(
            image, candidates, k, question, ocr_boxes, version="v2", second_slot="evidence_topk",
        )
    elif method == "qe_bops_pair_top8":
        selected, scores, all_scores, top_feats = _select_qe_bops_pair_top8(
            image, candidates, k, question, ocr_boxes, version="v2",
        )
    elif method == "qe_bops_safe_expand":
        selected, scores, all_scores, top_feats = _select_qe_bops_safe_expand(
            image, candidates, k, question, ocr_boxes, version="v2",
        )
    elif method == "qe_bops_node_pair":
        selected, scores, all_scores, node_meta = select_node_pair_patches(
            image, candidates, k, question, ocr_boxes,
        )
        top_feats = node_meta
    elif method == "qe_bops_table_pair":
        selected, scores, all_scores, table_meta = select_table_pair_patches(
            image, candidates, k, question, ocr_boxes, swap_margin=0.05,
        )
        top_feats = table_meta
    elif method == "qe_bops_entity_row":
        selected, scores, all_scores, entity_meta = select_entity_row_patches(
            image, candidates, k, question, ocr_boxes, swap_margin=0.05,
        )
        top_feats = entity_meta
    elif method == "qe_bops_no_question":
        selected, scores, all_scores, top_feats = _select_qe_bops(
            image, candidates, k, question, ocr_boxes,
            include_question=False, version="v1", mmr_lambda=mmr_lambda, use_mmr=True,
        )
    elif method in (
        "learned_logreg",
        "learned_lgbm_strict",
        "learned_lgbm_any",
        "learned_lgbm_combined",
        "learned_lgbm_qbops_hybrid",
    ):
        selected, scores, all_scores, top_feats = _select_learned_ranker(
            image, candidates, k, question, ocr_boxes, method,
            image_id=image_id,
            question_id=question_id,
            prefer_oof=prefer_oof,
        )
    elif method == "oracle":
        if not patch_labels:
            raise ValueError("oracle selection requires patch_labels from eval pipeline")
        selected = oracle_select_patches(candidates, patch_labels, k)
        all_scores = [float(lbl.get("label_confidence", 0.0)) for lbl in patch_labels]
        scores = [lbl["label_confidence"] for lbl in patch_labels[: len(selected)]]
    else:
        raise ValueError(f"Unknown method: {method}")

    patch_images = [crop_patch(image, p) for p in selected]
    meta = {
        "method": method,
        "num_candidates": len(candidates),
        "num_selected": len(selected),
        "pool_hash": ph,
        "candidate_scores": all_scores,
        "mmr_lambda": mmr_lambda,
    }
    if top_feats:
        if method in (
            "qe_bops_node_pair", "qe_bops_table_pair", "qe_bops_entity_row",
        ) or method.startswith("learned_") or method == "learned_logreg":
            meta.update({k: v for k, v in top_feats.items() if k not in meta})
        else:
            meta["top_features"] = top_feats
    return SelectionResult(method, selected, patch_images, scores, candidates, ph, meta)


def assert_same_pool_hash(a: SelectionResult, b: SelectionResult) -> None:
    if a.pool_hash_value != b.pool_hash_value:
        raise AssertionError(f"Pool hash mismatch: {a.method} vs {b.method}")
