#!/usr/bin/env python3
"""Evaluate learned rankers on coverage@K vs Q-BOPS baselines.

--oof       : use OOF patch scores (debug / image CV)
--held-out  : score held-out val images with final lgbm_*.txt models
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import yaml

from src.data.dataset_loader import iter_manifest
from src.features.ranker_features import FEATURE_KEYS, extract_ranker_features_for_patches
from src.metrics.coverage_eval import eval_selection
from src.preprocessing.patch_grid import Patch
from src.preprocessing.selectors import select_patches
from src.utils.image_io import load_image
from src.utils.logging_utils import log_section, setup_experiment_logging
from src.utils.ocr_cache import load_cached_ocr_boxes
from src.utils.paths import outputs_path, repo_path
from src.metrics.statistical_tests import bootstrap_ci


def _ndcg_at_k(relevances: list[float], k: int) -> float:
    rel = relevances[:k]
    if not rel or max(relevances) <= 0:
        return 0.0
    dcg = sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(rel))
    ideal = sorted(relevances, reverse=True)[:k]
    idcg = sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


def _mrr(relevances: list[float]) -> float:
    for i, r in enumerate(relevances):
        if r > 0:
            return 1.0 / (i + 1)
    return 0.0


def _labels_for_image(labels_df: pd.DataFrame, iid: str) -> list[dict]:
    sub = labels_df[labels_df["image_id"] == iid]
    return [r.to_dict() for _, r in sub.iterrows()]


def _load_oof(tag: str) -> pd.DataFrame | None:
    paths = [
        outputs_path("ranker", f"oof_scores_strict_{tag}.parquet"),
        outputs_path("ranker", "oof_scores_strict.parquet"),
    ]
    strict_path = next((p for p in paths if p.exists()), None)
    any_paths = [
        outputs_path("ranker", f"oof_scores_any_{tag}.parquet"),
        outputs_path("ranker", "oof_scores_any.parquet"),
    ]
    any_path = next((p for p in any_paths if p.exists()), None)
    if not strict_path and not any_path:
        return None
    frames = []
    if strict_path:
        frames.append(pd.read_parquet(strict_path))
    if any_path:
        frames.append(pd.read_parquet(any_path))
    if len(frames) == 1:
        return frames[0]
    return frames[0].merge(
        frames[1][["image_id", "question_id", "patch_index", "score_any"]],
        on=["image_id", "question_id", "patch_index"],
        how="outer",
        suffixes=("", "_dup"),
    )


def _cfg_weights() -> dict:
    path = repo_path("configs", "qe_bops.yaml")
    if not path.exists():
        return {"strict": 0.6, "any": 0.4, "learned": 0.5, "qbops": 0.5}
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    comb = cfg.get("ranker", {}).get("combined", {})
    hyb = cfg.get("ranker", {}).get("qbops_hybrid", {})
    return {
        "strict": float(comb.get("strict_weight", 0.6)),
        "any": float(comb.get("any_weight", 0.4)),
        "learned": float(hyb.get("learned_weight", 0.5)),
        "qbops": float(hyb.get("qbops_weight", 0.5)),
    }


def _attach_logreg_scores(score_df: pd.DataFrame, dataset: pd.DataFrame | None) -> pd.DataFrame:
    """Add logreg_strict / logreg_any columns from checkpoints when dataset features exist."""
    import joblib

    if dataset is None:
        return score_df
    merge_cols = ["image_id", "question_id", "patch_index"]
    feats = dataset[merge_cols + FEATURE_KEYS].copy()
    out = score_df.merge(feats, on=merge_cols, how="left", suffixes=("", "_f"))
    X = out[FEATURE_KEYS].fillna(0.0).to_numpy(dtype=float)
    for short in ("strict", "any"):
        path = outputs_path("checkpoints", f"logreg_{short}.pkl")
        if not path.exists():
            continue
        blob = joblib.load(path)
        Xs = blob["scaler"].transform(X)
        out[f"logreg_{short}"] = blob["model"].predict_proba(Xs)[:, 1]
    return out


def _score_column(df: pd.DataFrame, model: str, weights: dict) -> pd.Series:
    s_strict = df["score_strict"] if "score_strict" in df.columns else pd.Series(0.0, index=df.index)
    s_any = df["score_any"] if "score_any" in df.columns else pd.Series(0.0, index=df.index)
    q = df["q_bops_score"] if "q_bops_score" in df.columns else pd.Series(0.0, index=df.index)
    if model == "logreg_strict":
        return df["logreg_strict"] if "logreg_strict" in df.columns else s_strict
    if model == "logreg_any":
        return df["logreg_any"] if "logreg_any" in df.columns else s_any
    if model == "lgbm_strict":
        return s_strict
    if model == "lgbm_any":
        return s_any
    if model == "lgbm_combined":
        return weights["strict"] * s_strict + weights["any"] * s_any
    if model == "lgbm_qbops_hybrid":
        learned = weights["strict"] * s_strict + weights["any"] * s_any

        def _norm(g: pd.Series) -> pd.Series:
            lo, hi = g.min(), g.max()
            if hi <= lo:
                return pd.Series(0.0, index=g.index)
            return (g - lo) / (hi - lo)

        learned_n = learned.groupby(df["question_id"]).transform(_norm)
        q_n = q.groupby(df["question_id"]).transform(_norm)
        return weights["learned"] * learned_n + weights["qbops"] * q_n
    raise ValueError(f"Unknown model: {model}")


def _predict_held_out(dataset: pd.DataFrame, val_images: set[str], models: list[str]) -> pd.DataFrame:
    """Score held-out rows with final LightGBM models (+ logreg if requested)."""
    import lightgbm as lgb
    import joblib

    sub = dataset[dataset["image_id"].isin(val_images)].copy()
    if sub.empty:
        raise SystemExit("No held-out val rows in dataset")

    X = sub[FEATURE_KEYS].to_numpy(dtype=float)
    if any(m.startswith("lgbm") for m in models):
        for short, col in (("strict", "score_strict"), ("any", "score_any")):
            path = outputs_path("checkpoints", f"lgbm_{short}.txt")
            if path.exists():
                booster = lgb.Booster(model_file=str(path))
                sub[col] = booster.predict(X)
            elif col not in sub.columns:
                sub[col] = 0.0
    if any(m.startswith("logreg") for m in models):
        for short, col in (("strict", "score_strict"), ("any", "score_any")):
            path = outputs_path("checkpoints", f"logreg_{short}.pkl")
            if not path.exists():
                continue
            blob = joblib.load(path)
            Xs = blob["scaler"].transform(X)
            probs = blob["model"].predict_proba(Xs)[:, 1]
            # Only overwrite if this is the requested logreg target column
            if f"logreg_{short}" in models or "logreg_strict" in models or "logreg_any" in models:
                if short == "strict" or "score_strict" not in sub.columns or sub["score_strict"].sum() == 0:
                    if "lgbm" not in "".join(models) or short == "strict":
                        pass
            # Prefer keep lgbm scores; store logreg separately if needed
            sub[f"logreg_{short}"] = probs
            if f"score_{short}" not in sub.columns:
                sub[f"score_{short}"] = probs
    return sub


def main() -> None:
    parser = argparse.ArgumentParser(description="Learned ranker coverage@K eval.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--models", default="lgbm_strict,lgbm_any,lgbm_combined,lgbm_qbops_hybrid")
    parser.add_argument("--baselines", default="bops_qa_fair_pool,qe_bops_v2,bm25_only,bops_fair_pool")
    parser.add_argument("--k", default="1,2,4,8")
    parser.add_argument("--oof", action="store_true")
    parser.add_argument("--held-out", action="store_true")
    parser.add_argument("--dataset-tag", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logger = setup_experiment_logging("eval_learned_ranker")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    baselines = [m.strip() for m in args.baselines.split(",") if m.strip()]
    k_values = [int(x) for x in args.k.split(",") if x.strip()]
    tag = args.dataset_tag or args.manifest.stem.replace("docvqa_", "")
    weights = _cfg_weights()
    log_section(logger, f"Learned coverage | models={models} k={k_values} oof={args.oof} held_out={args.held_out}")

    labels_df = pd.read_parquet(outputs_path("labels", "patch_labels.parquet"))
    records = list(iter_manifest(args.manifest))
    iid_of = lambda r: r.get("image_id") or r.get("doc_id") or r.get("question_id", "")

    if args.oof:
        score_df = _load_oof(tag)
        if score_df is None:
            raise SystemExit("Missing OOF score files. Train with --cv 5 first.")
        ds_path = outputs_path("ranker", f"ranker_dataset_{tag}.parquet")
        if not ds_path.exists():
            ds_path = outputs_path("ranker", "ranker_dataset.parquet")
        dataset = pd.read_parquet(ds_path) if ds_path.exists() else None
        if any(m.startswith("logreg") for m in models):
            score_df = _attach_logreg_scores(score_df, dataset)
    elif args.held_out:
        ds_path = outputs_path("ranker", f"ranker_dataset_{tag}.parquet")
        if not ds_path.exists():
            ds_path = outputs_path("ranker", "ranker_dataset.parquet")
        dataset = pd.read_parquet(ds_path)
        val_images = set(dataset.loc[dataset["split"] == "val", "image_id"].unique())
        score_df = _predict_held_out(dataset, val_images, models)
        # Restrict eval to held-out images
        records = [r for r in records if iid_of(r) in val_images]
    else:
        raise SystemExit("Specify --oof or --held-out")

    # Ensure required score columns
    if "score_strict" not in score_df.columns:
        score_df["score_strict"] = 0.0
    if "score_any" not in score_df.columns:
        score_df["score_any"] = score_df.get("score_strict", 0.0)
    if "q_bops_score" not in score_df.columns:
        score_df["q_bops_score"] = 0.0

    per_q_rows: list[dict] = []
    summary_rows: list[dict] = []

    # ---- Learned models from scores ----
    for model in models:
        score_df = score_df.copy()
        score_df["_sel"] = _score_column(score_df, model, weights)
        for hk in k_values:
            samples = []
            for rec in records:
                iid = iid_of(rec)
                qid = str(rec.get("question_id", ""))
                patch_labels = _labels_for_image(labels_df, iid)
                labels_by_index = {int(l.get("patch_index", l.get("index", -1))): l for l in patch_labels}
                sub = score_df[(score_df["image_id"] == iid)]
                if qid and "question_id" in sub.columns:
                    qsub = sub[sub["question_id"].astype(str) == qid]
                    if not qsub.empty:
                        sub = qsub
                if sub.empty:
                    continue
                ranked = sub.sort_values("_sel", ascending=False)
                top = ranked.head(hk)
                sel_idx = set(int(x) for x in top["patch_index"].tolist())
                # texts unused for strict/any from labels; OCR exact needs texts
                texts = []
                patch_path = outputs_path("ocr", "patches", f"{iid}.json")
                if patch_path.exists():
                    import json as _json
                    with open(patch_path, encoding="utf-8") as f:
                        pdata = _json.load(f)
                    texts = [p.get("text", "") for p in pdata["patches"] if p.get("index") in sel_idx]
                answers = rec.get("answers") or rec.get("answer") or []
                if isinstance(answers, str):
                    answers = [answers]
                ranked_indices = [int(r.patch_index) for r in ranked.itertuples()]
                ev = eval_selection(patch_labels, sel_idx, list(answers), texts, ranked_indices)

                # Ranking metrics over full candidate list
                rel_strict = []
                rel_any = []
                for r in ranked.itertuples():
                    lab = labels_by_index.get(int(r.patch_index), {})
                    s = 1.0 if (lab.get("label_exact_patch_ocr") or lab.get("label_fullpage_box_overlap")) else 0.0
                    a = 1.0 if lab.get("label_positive") else 0.0
                    rel_strict.append(s)
                    rel_any.append(a)

                row = {
                    "image_id": iid,
                    "question_id": qid,
                    "method": model,
                    "k": hk,
                    "evidence_strict": float(ev["evidence_strict"]),
                    "evidence_any": float(ev["evidence_any"]),
                    "evidence_coverage": float(ev["evidence_any"]),
                    "ocr_exact_coverage": float(ev["ocr_exact_coverage"]),
                    "ndcg_strict": _ndcg_at_k(rel_strict, hk),
                    "ndcg_any": _ndcg_at_k(rel_any, hk),
                    "mrr_strict": _mrr(rel_strict),
                    "mrr_any": _mrr(rel_any),
                }
                samples.append(row)
                per_q_rows.append(row)

            if not samples:
                continue
            summary = {
                "method": model,
                "k": hk,
                "n": len(samples),
                "evidence_strict": sum(s["evidence_strict"] for s in samples) / len(samples),
                "evidence_any": sum(s["evidence_any"] for s in samples) / len(samples),
                "ocr_exact_coverage": sum(s["ocr_exact_coverage"] for s in samples) / len(samples),
                "ndcg_strict": sum(s["ndcg_strict"] for s in samples) / len(samples),
                "ndcg_any": sum(s["ndcg_any"] for s in samples) / len(samples),
                "mrr_strict": sum(s["mrr_strict"] for s in samples) / len(samples),
                "mrr_any": sum(s["mrr_any"] for s in samples) / len(samples),
            }
            summary_rows.append(summary)
            logger.info(
                "%s | K=%d strict=%.3f any=%.3f ndcg_s=%.3f mrr_s=%.3f",
                model, hk, summary["evidence_strict"], summary["evidence_any"],
                summary["ndcg_strict"], summary["mrr_strict"],
            )

    # ---- Baselines via selectors (score once at max K; reselect per K) ----
    max_k = max(k_values) if k_values else 2
    for method in baselines:
        samples_by_k: dict[int, list[dict]] = {hk: [] for hk in k_values}
        for rec in records:
            iid = iid_of(rec)
            q = rec.get("question", "")
            answers = rec.get("answers") or rec.get("answer") or []
            if isinstance(answers, str):
                answers = [answers]
            try:
                image = load_image(rec["image_path"])
            except Exception:
                continue
            boxes = load_cached_ocr_boxes(iid) or []
            patch_labels = _labels_for_image(labels_df, iid)
            labels_by_index = {int(l.get("patch_index", l.get("index", -1))): l for l in patch_labels}
            sel = select_patches(image, method, max_k, q, boxes)
            all_scores = sel.meta.get("candidate_scores", [0.0] * len(sel.candidate_pool))
            from src.preprocessing.mmr_select import mmr_select

            # Preload patch texts
            texts_by_idx: dict[int, str] = {}
            patch_path = outputs_path("ocr", "patches", f"{iid}.json")
            if patch_path.exists():
                with open(patch_path, encoding="utf-8") as f:
                    pdata = json.load(f)
                for p in pdata["patches"]:
                    texts_by_idx[int(p.get("index", -1))] = p.get("text", "")

            for hk in k_values:
                if method in ("bops_qa_fair_pool", "qe_bops_v2", "bops_fair_pool") or "qe_bops" in method:
                    # Re-run selection policy at this K from the same pool/scores
                    if method == "bops_qa_fair_pool":
                        selected = mmr_select(sel.candidate_pool, list(all_scores), hk, lambda_=0.5)
                    elif method == "bops_fair_pool":
                        ranked = sorted(zip(sel.candidate_pool, all_scores), key=lambda x: x[1], reverse=True)
                        selected = [p for p, _ in ranked[:hk]]
                    elif method == "qe_bops_v2":
                        selected = mmr_select(sel.candidate_pool, list(all_scores), hk, lambda_=0.75)
                    else:
                        selected = mmr_select(sel.candidate_pool, list(all_scores), hk, lambda_=0.5)
                else:
                    ranked = sorted(zip(sel.candidate_pool, all_scores), key=lambda x: x[1], reverse=True)
                    selected = [p for p, _ in ranked[:hk]]

                sel_idx = {p.index for p in selected}
                texts = [texts_by_idx.get(i, "") for i in sel_idx]
                ranked_full = sorted(zip(sel.candidate_pool, all_scores), key=lambda x: x[1], reverse=True)
                ranked_indices = [p.index for p, _ in ranked_full]
                ev = eval_selection(patch_labels, sel_idx, list(answers), texts, ranked_indices)
                rel_strict, rel_any = [], []
                ordered_idx = [p.index for p in selected] + [i for i in ranked_indices if i not in sel_idx]
                for idx in ordered_idx:
                    lab = labels_by_index.get(idx, {})
                    rel_strict.append(1.0 if (lab.get("label_exact_patch_ocr") or lab.get("label_fullpage_box_overlap")) else 0.0)
                    rel_any.append(1.0 if lab.get("label_positive") else 0.0)
                row = {
                    "image_id": iid,
                    "question_id": rec.get("question_id", ""),
                    "method": method,
                    "k": hk,
                    "evidence_strict": float(ev["evidence_strict"]),
                    "evidence_any": float(ev["evidence_any"]),
                    "evidence_coverage": float(ev["evidence_any"]),
                    "ocr_exact_coverage": float(ev["ocr_exact_coverage"]),
                    "ndcg_strict": _ndcg_at_k(rel_strict, hk),
                    "ndcg_any": _ndcg_at_k(rel_any, hk),
                    "mrr_strict": _mrr(rel_strict),
                    "mrr_any": _mrr(rel_any),
                }
                samples_by_k[hk].append(row)
                per_q_rows.append(row)

        for hk in k_values:
            samples = samples_by_k[hk]
            if not samples:
                continue
            summary = {
                "method": method,
                "k": hk,
                "n": len(samples),
                "evidence_strict": sum(s["evidence_strict"] for s in samples) / len(samples),
                "evidence_any": sum(s["evidence_any"] for s in samples) / len(samples),
                "ocr_exact_coverage": sum(s["ocr_exact_coverage"] for s in samples) / len(samples),
                "ndcg_strict": sum(s["ndcg_strict"] for s in samples) / len(samples),
                "ndcg_any": sum(s["ndcg_any"] for s in samples) / len(samples),
                "mrr_strict": sum(s["mrr_strict"] for s in samples) / len(samples),
                "mrr_any": sum(s["mrr_any"] for s in samples) / len(samples),
            }
            summary_rows.append(summary)
            logger.info(
                "%s | K=%d strict=%.3f any=%.3f",
                method, hk, summary["evidence_strict"], summary["evidence_any"],
            )

    tag_suffix = f"_{tag}" if tag else ""
    out_sum = outputs_path("metrics", f"learned_coverage_by_method{tag_suffix}.csv")
    out_pq = outputs_path("metrics", f"learned_coverage_per_question{tag_suffix}.csv")
    # Also write untagged copies for gate defaults when tag is set
    out_sum_default = outputs_path("metrics", "learned_coverage_by_method.csv")
    out_pq_default = outputs_path("metrics", "learned_coverage_per_question.csv")
    fields = [
        "method", "k", "evidence_strict", "evidence_any", "ocr_exact_coverage",
        "ndcg_strict", "ndcg_any", "mrr_strict", "mrr_any", "n",
    ]
    with open(out_sum, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary_rows)
    pd.DataFrame(per_q_rows).to_csv(out_pq, index=False)
    if tag_suffix:
        with open(out_sum_default, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(summary_rows)
        pd.DataFrame(per_q_rows).to_csv(out_pq_default, index=False)
    logger.info("Wrote %s and %s", out_sum, out_pq)

    # Bootstrap best learned vs Q-BOPS at each K
    boot_results = []
    pq = pd.DataFrame(per_q_rows)
    for model in models:
        for hk in k_values:
            for metric in ("evidence_strict", "evidence_any"):
                a = pq[(pq["method"] == model) & (pq["k"] == hk)]
                b = pq[(pq["method"] == "bops_qa_fair_pool") & (pq["k"] == hk)]
                if a.empty or b.empty:
                    continue
                merged = a.merge(b, on="image_id", suffixes=("_m", "_b"))
                diffs = (merged[f"{metric}_m"].astype(float) - merged[f"{metric}_b"].astype(float)).tolist()
                mean, lo, hi = bootstrap_ci(diffs) if diffs else (0.0, 0.0, 0.0)
                boot_results.append({
                    "comparison": f"{model}_vs_bops_qa_fair_pool",
                    "metric": metric,
                    "k": hk,
                    "mean_diff": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_pairs": len(diffs),
                })
    boot_path = outputs_path("metrics", "coverage_bootstrap_ci_learned.json")
    with open(boot_path, "w", encoding="utf-8") as f:
        json.dump(boot_results, f, indent=2)
    logger.info("Wrote %s", boot_path)

    # Gate summary at K=2 and K=4 for each learned model
    gate_lines = []
    for model in models:
        for hk in (2, 4):
            mrow = next((r for r in summary_rows if r["method"] == model and r["k"] == hk), None)
            brow = next((r for r in summary_rows if r["method"] == "bops_qa_fair_pool" and r["k"] == hk), None)
            if not mrow or not brow:
                continue
            strict_ok = mrow["evidence_strict"] > brow["evidence_strict"]
            any_ok = mrow["evidence_any"] >= brow["evidence_any"]
            passed = strict_ok and any_ok
            gate_lines.append({
                "model": model,
                "k": hk,
                "learned_strict": mrow["evidence_strict"],
                "qbops_strict": brow["evidence_strict"],
                "learned_any": mrow["evidence_any"],
                "qbops_any": brow["evidence_any"],
                "passed": passed,
            })
            logger.info(
                "GATE %s@%d: %s (strict %.3f vs %.3f, any %.3f vs %.3f)",
                model, hk, "PASS" if passed else "FAIL",
                mrow["evidence_strict"], brow["evidence_strict"],
                mrow["evidence_any"], brow["evidence_any"],
            )
    with open(outputs_path("gates", "learned_ranker_gate.json"), "w", encoding="utf-8") as f:
        json.dump({"results": gate_lines, "any_passed": any(g["passed"] for g in gate_lines)}, f, indent=2)


if __name__ == "__main__":
    main()
