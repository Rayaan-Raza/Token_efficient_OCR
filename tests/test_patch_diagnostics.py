"""Unit tests for post-hoc patch diagnostics (:mod:`src.vlm.patch_diagnostics`)."""

from PIL import Image

from src.preprocessing.bops import run_bops
from src.vlm.patch_diagnostics import _answer_in_text, compute_patch_diagnostics


def test_answer_in_text_substring():
    assert _answer_in_text(["University of California"], "the university of california campus") is True
    assert _answer_in_text(["ITC Limited"], "some other text") is False


def test_bops_selection_does_not_use_answers():
    """run_bops must not accept answers; diagnostics are post-hoc only."""
    import inspect

    sig = inspect.signature(run_bops)
    assert "answer" not in sig.parameters
    assert "ground_truth" not in sig.parameters


def test_compute_patch_diagnostics_overview_only():
    img = Image.new("RGB", (400, 300), color=(255, 255, 255))
    result = run_bops(img, 0, mode="overview_only")
    diag = compute_patch_diagnostics(
        img, "test_img", "overview_only", 0, result, ["hello"]
    )
    assert "selected_patch_coords" in diag
    assert diag["mean_patch_score"] == 0.0
    assert diag["num_ocr_boxes_selected"] == 0
