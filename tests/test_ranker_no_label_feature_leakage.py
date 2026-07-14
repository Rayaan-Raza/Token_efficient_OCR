"""Ensure ranker feature matrices never include labels or gold answers."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.features.ranker_features import (
    FEATURE_KEYS,
    FORBIDDEN_FEATURE_COLUMNS,
    assert_no_feature_leakage,
)
from src.utils.paths import outputs_path


def test_feature_keys_exclude_forbidden():
    overlap = set(FEATURE_KEYS) & FORBIDDEN_FEATURE_COLUMNS
    assert not overlap, f"FEATURE_KEYS contains forbidden columns: {overlap}"


def test_assert_no_feature_leakage_raises():
    with pytest.raises(ValueError):
        assert_no_feature_leakage(["bm25", "strict_positive", "q_bops_score"])


def test_assert_no_feature_leakage_ok():
    assert_no_feature_leakage(FEATURE_KEYS)


def test_ranker_dataset_schema_no_feature_leakage():
    schema_path = outputs_path("ranker", "ranker_dataset_schema.json")
    if not schema_path.exists():
        pytest.skip("ranker dataset not built yet")
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)
    keys = schema.get("feature_keys", [])
    overlap = set(keys) & FORBIDDEN_FEATURE_COLUMNS
    assert not overlap


def test_ranker_dataset_parquet_features_clean():
    path = outputs_path("ranker", "ranker_dataset.parquet")
    if not path.exists():
        pytest.skip("ranker dataset not built yet")
    df = pd.read_parquet(path)
    feature_cols = [c for c in FEATURE_KEYS if c in df.columns]
    assert feature_cols, "expected FEATURE_KEYS columns in dataset"
    overlap = set(feature_cols) & FORBIDDEN_FEATURE_COLUMNS
    assert not overlap
    # Targets may exist as columns but must not be in FEATURE_KEYS
    for col in ("strict_positive", "any_positive", "label_confidence"):
        if col in df.columns:
            assert col not in FEATURE_KEYS
