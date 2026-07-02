"""Tests for question-aware patch scoring."""

from PIL import Image

from src.preprocessing.patch_grid import Patch
from src.preprocessing.patch_scoring_qa import (
    question_token_overlap_score,
    score_patch_question_aware,
)


def test_question_overlap_prefers_matching_patch():
    img = Image.new("RGB", (400, 400), color=(255, 255, 255))
    boxes = [
        {"box": [[10, 10], [100, 10], [100, 40], [10, 40]], "text": "Chennai office", "confidence": 0.9},
        {"box": [[200, 200], [300, 200], [300, 240], [200, 240]], "text": "random text", "confidence": 0.8},
    ]
    p_match = Patch(x=0, y=0, w=120, h=60, index=0)
    p_other = Patch(x=180, y=180, w=140, h=80, index=1)
    q = "Where is Chennai located?"
    assert question_token_overlap_score(p_match, boxes, q) > question_token_overlap_score(p_other, boxes, q)
    assert score_patch_question_aware(img, p_match, boxes, q) > score_patch_question_aware(
        img, p_other, boxes, q
    )
