"""Hierarchical label rules."""

from __future__ import annotations

from src.metrics.answer_coverage import compute_patch_labels
from src.preprocessing.patch_grid import Patch


def test_fullpage_box_overlap_center_in_patch():
    patch = Patch(0, 0, 200, 200, 0)
    answer_boxes = [{"box": [[50, 50], [80, 50], [80, 70], [50, 70]], "text": "2020"}]
    labels = compute_patch_labels("", patch, ["2020"], answer_boxes)
    assert labels["label_fullpage_box_overlap"] is True
    assert labels["label_confidence"] >= 0.9
