"""Tests for QE-BOPS node-pair evidence retrieval."""

from __future__ import annotations

from PIL import Image

from src.preprocessing.patch_grid import Patch
from src.preprocessing.qe_bops_node_pair import (
    enumerate_node_pairs,
    late_interaction_score,
    select_node_pair_patches,
)
from src.features.ocr_layout_graph import build_ocr_layout_graph


def _sample_boxes():
    return [
        {"box": [[10, 10], [80, 10], [80, 30], [10, 30]], "text": "Name:", "confidence": 0.95},
        {"box": [[90, 10], [180, 10], [180, 30], [90, 30]], "text": "John Smith", "confidence": 0.92},
        {"box": [[10, 50], [120, 50], [120, 70], [10, 70]], "text": "Date 2024-01-15", "confidence": 0.90},
    ]


def test_late_interaction_exact_token():
    assert late_interaction_score("What is the name?", "John Smith") >= 0.0


def test_enumerate_node_pairs_finds_row_pair():
    graph = build_ocr_layout_graph(_sample_boxes())
    pairs = enumerate_node_pairs(graph, "What is the name?")
    assert pairs
    assert any(p.relation in ("same_row", "nearby_right") for p in pairs)


def test_select_node_pair_returns_k_patches():
    image = Image.new("RGB", (256, 256), color=(255, 255, 255))
    boxes = _sample_boxes()
    pool = [Patch(0, 0, 128, 128, 0), Patch(64, 0, 128, 128, 1), Patch(0, 64, 128, 128, 2)]
    selected, scores, _, meta = select_node_pair_patches(image, pool, 2, "What is the name?", boxes)
    assert len(selected) == 2
    assert len(scores) == 2
    assert meta.get("node_pair_count", 0) >= 0
