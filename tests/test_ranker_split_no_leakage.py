"""Ranker split must not share image_ids between train and val."""

from __future__ import annotations

import json

from src.utils.paths import outputs_path


def test_ranker_split_disjoint_images():
    path = outputs_path("gates", "docvqa_ranker_split.json")
    if not path.exists():
        return  # manifests not built in CI yet
    with open(path, encoding="utf-8") as f:
        split = json.load(f)
    train = set(split["train_image_ids"])
    val = set(split["val_image_ids"])
    assert train.isdisjoint(val)
