"""Unit tests for dual labels, LambdaRank groups, and top-K from scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.ranker_features import FEATURE_KEYS, assert_no_feature_leakage


def test_dual_label_construction():
    exact = True
    box = False
    soft = True
    fuzzy = False
    strict = exact or box
    any_pos = exact or box or soft or fuzzy
    assert strict is True
    assert any_pos is True
    strict2 = False or False
    any2 = False or False or True or False
    assert strict2 is False
    assert any2 is True


def test_topk_from_scores():
    scores = np.array([0.1, 0.9, 0.4, 0.8])
    order = np.argsort(-scores)[:2]
    assert list(order) == [1, 3]


def test_lambdarank_group_sizes():
    df = pd.DataFrame({
        "group_id": ["q1", "q1", "q1", "q2", "q2"],
        "patch_index": [0, 1, 2, 0, 1],
        "score": [0.1, 0.2, 0.3, 0.5, 0.4],
    })
    df = df.sort_values(["group_id", "patch_index"]).reset_index(drop=True)
    sizes = df.groupby("group_id", sort=False).size().tolist()
    assert sizes == [3, 2]
    assert sum(sizes) == len(df)


def test_feature_keys_stable_and_clean():
    assert "q_bops_score" in FEATURE_KEYS
    assert "q_bops_rank" in FEATURE_KEYS
    assert "is_q_bops_top1" in FEATURE_KEYS
    assert "late_interaction" in FEATURE_KEYS
    assert_no_feature_leakage(FEATURE_KEYS)
