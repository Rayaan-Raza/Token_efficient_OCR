"""Tests for evidence vs OCR-exact coverage helpers."""

from src.metrics.answer_coverage import (
    evidence_in_selected,
    mean_rank_of_first_positive,
    pool_reachability_rates,
    rank_candidates_by_score,
)
from src.preprocessing.patch_grid import Patch


def test_evidence_in_selected():
    labels = [
        {"patch_index": 0, "label_positive": False},
        {"patch_index": 1, "label_positive": True},
    ]
    assert evidence_in_selected(labels, {0}) is False
    assert evidence_in_selected(labels, {1}) is True


def test_mean_rank_of_first_positive():
    labels = {
        0: {"label_positive": False},
        2: {"label_positive": True},
        5: {"label_positive": True},
    }
    assert mean_rank_of_first_positive([0, 1, 2, 3], labels) == 3.0


def test_pool_reachability_rates():
    labels = [
        {"label_positive": True, "label_exact_patch_ocr": False, "label_fullpage_box_overlap": True,
         "label_soft_token_overlap": False, "label_fuzzy_match": False},
    ]
    rates = pool_reachability_rates(labels)
    assert rates["candidate_evidence_reachability"] is True
    assert rates["candidate_ocr_exact_reachability"] is False


def test_rank_candidates_by_score():
    patches = [Patch(0, 0, 1, 1, 0), Patch(1, 1, 1, 1, 1)]
    ranked = rank_candidates_by_score(patches, [0.1, 0.9])
    assert ranked == [1, 0]
