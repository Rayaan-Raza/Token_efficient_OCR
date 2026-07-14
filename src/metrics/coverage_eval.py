"""Shared coverage evaluation helpers for G3 diagnostics and selector eval."""

from __future__ import annotations

from typing import Any

from src.metrics.answer_coverage import (
    _label_index,
    answer_in_selected_patches,
    label_field_in_selected,
    mean_rank_of_first_positive,
    rank_candidates_by_score,
)
from src.preprocessing.patch_grid import Patch


def patch_evidence_tiers(lbl: dict[str, Any]) -> dict[str, bool]:
    strict = bool(lbl.get("label_exact_patch_ocr")) or bool(lbl.get("label_fullpage_box_overlap"))
    soft_only = bool(lbl.get("label_soft_token_overlap")) or bool(lbl.get("label_fuzzy_match"))
    return {
        "strict": strict,
        "soft": soft_only and not strict,
        "any": strict or soft_only,
    }


def tier_in_selected(
    labels: list[dict[str, Any]],
    selected_indices: set[int],
    tier: str,
) -> bool:
    for lbl in labels:
        if _label_index(lbl) not in selected_indices:
            continue
        tiers = patch_evidence_tiers(lbl)
        if tier == "strict" and tiers["strict"]:
            return True
        if tier == "soft" and tiers["soft"]:
            return True
        if tier == "any" and tiers["any"]:
            return True
    return False


def reciprocal_rank(
    ranked_indices: list[int],
    labels_by_index: dict[int, dict[str, Any]],
    tier: str = "any",
) -> float:
    for rank, idx in enumerate(ranked_indices, start=1):
        lbl = labels_by_index.get(idx)
        if not lbl:
            continue
        tiers = patch_evidence_tiers(lbl)
        if tier == "strict" and tiers["strict"]:
            return 1.0 / rank
        if tier == "any" and tiers["any"]:
            return 1.0 / rank
    return 0.0


def first_positive_rank(
    ranked_indices: list[int],
    labels_by_index: dict[int, dict[str, Any]],
    tier: str = "any",
) -> float | None:
    for rank, idx in enumerate(ranked_indices, start=1):
        lbl = labels_by_index.get(idx)
        if not lbl:
            continue
        tiers = patch_evidence_tiers(lbl)
        if tier == "strict" and tiers["strict"]:
            return float(rank)
        if tier == "any" and tiers["any"]:
            return float(rank)
    return None


def coverage_at_k_from_ranked(
    ranked_indices: list[int],
    labels_by_index: dict[int, dict[str, Any]],
    k: int,
    tier: str = "any",
) -> bool:
    top = set(ranked_indices[:k])
    labels = list(labels_by_index.values())
    return tier_in_selected(labels, top, tier)


def eval_selection(
    labels: list[dict[str, Any]],
    selected_indices: set[int],
    answers: list[str],
    patch_texts: list[str],
    ranked_indices: list[int] | None = None,
) -> dict[str, Any]:
    labels_by_index = {_label_index(l): l for l in labels}
    ranked = ranked_indices or list(selected_indices)

    pos_ranks = [first_positive_rank(ranked, labels_by_index, t) for t in ("strict", "any")]
    return {
        "evidence_strict": tier_in_selected(labels, selected_indices, "strict"),
        "evidence_soft": tier_in_selected(labels, selected_indices, "soft"),
        "evidence_any": tier_in_selected(labels, selected_indices, "any"),
        "ocr_exact_coverage": answer_in_selected_patches(answers, patch_texts),
        "box_overlap_coverage": label_field_in_selected(
            labels, selected_indices, "label_fullpage_box_overlap"
        ),
        "soft_token_coverage": label_field_in_selected(
            labels, selected_indices, "label_soft_token_overlap"
        ),
        "fuzzy_coverage": label_field_in_selected(labels, selected_indices, "label_fuzzy_match"),
        "mean_rank_first_positive_strict": pos_ranks[0],
        "mean_rank_first_positive_any": pos_ranks[1],
        "mrr_strict": reciprocal_rank(ranked, labels_by_index, "strict"),
        "mrr_any": reciprocal_rank(ranked, labels_by_index, "any"),
        "coverage_at_1_strict": coverage_at_k_from_ranked(ranked, labels_by_index, 1, "strict"),
        "coverage_at_1_any": coverage_at_k_from_ranked(ranked, labels_by_index, 1, "any"),
        "coverage_at_2_strict": coverage_at_k_from_ranked(ranked, labels_by_index, 2, "strict"),
        "coverage_at_2_any": coverage_at_k_from_ranked(ranked, labels_by_index, 2, "any"),
        "coverage_at_4_strict": coverage_at_k_from_ranked(ranked, labels_by_index, 4, "strict"),
        "coverage_at_4_any": coverage_at_k_from_ranked(ranked, labels_by_index, 4, "any"),
        "coverage_at_8_strict": coverage_at_k_from_ranked(ranked, labels_by_index, 8, "strict"),
        "coverage_at_8_any": coverage_at_k_from_ranked(ranked, labels_by_index, 8, "any"),
    }


def positive_patch_stats(labels: list[dict[str, Any]], num_candidates: int) -> dict[str, float]:
    n_pos_any = sum(1 for l in labels if patch_evidence_tiers(l)["any"])
    n_pos_strict = sum(1 for l in labels if patch_evidence_tiers(l)["strict"])
    n_pos_soft_only = sum(1 for l in labels if patch_evidence_tiers(l)["soft"])
    return {
        "num_candidates": float(num_candidates),
        "num_positive_any": float(n_pos_any),
        "num_positive_strict": float(n_pos_strict),
        "num_positive_soft_only": float(n_pos_soft_only),
        "positive_patch_fraction_any": n_pos_any / max(1, num_candidates),
        "positive_patch_fraction_strict": n_pos_strict / max(1, num_candidates),
    }
