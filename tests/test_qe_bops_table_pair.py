"""Tests for QE-BOPS table-pair selection."""

from __future__ import annotations

from PIL import Image

from src.preprocessing.patch_grid import Patch
from src.preprocessing.qe_bops_table_pair import cluster_ocr_rows, select_table_pair_patches


def _table_boxes():
    return [
        {"box": [[10, 10], [60, 10], [60, 28], [10, 28]], "text": "Name", "confidence": 0.9},
        {"box": [[70, 10], [150, 10], [150, 28], [70, 28]], "text": "Alice", "confidence": 0.92},
        {"box": [[10, 35], [60, 35], [60, 53], [10, 53]], "text": "Age", "confidence": 0.9},
        {"box": [[70, 35], [100, 35], [100, 53], [70, 53]], "text": "42", "confidence": 0.88},
    ]


def test_cluster_rows_groups_same_y():
    rows = cluster_ocr_rows(_table_boxes(), "What is the name?")
    assert len(rows) == 2
    assert any(r.row_type in ("table_like_row", "question_row") for r in rows)


def test_select_table_pair_k2():
    image = Image.new("RGB", (200, 80), color=(255, 255, 255))
    boxes = _table_boxes()
    pool = [
        Patch(0, 0, 100, 40, 0),
        Patch(60, 0, 100, 40, 1),
        Patch(0, 30, 100, 40, 2),
    ]
    selected, scores, _, meta = select_table_pair_patches(
        image, pool, 2, "What is the name?", boxes,
    )
    assert len(selected) == 2
    assert meta.get("slot1_source") == "q_bops_anchor"
