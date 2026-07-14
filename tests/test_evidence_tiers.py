"""Tests for tiered evidence coverage."""

from src.metrics.coverage_eval import patch_evidence_tiers, tier_in_selected


def test_strict_excludes_soft_only():
    lbl = {
        "patch_index": 0,
        "label_exact_patch_ocr": False,
        "label_fullpage_box_overlap": False,
        "label_soft_token_overlap": True,
        "label_fuzzy_match": False,
    }
    tiers = patch_evidence_tiers(lbl)
    assert tiers["strict"] is False
    assert tiers["soft"] is True
    assert tiers["any"] is True
    assert tier_in_selected([lbl], {0}, "strict") is False
    assert tier_in_selected([lbl], {0}, "any") is True
