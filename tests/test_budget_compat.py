"""Tests for method×budget compatibility rules."""

from src.utils.budget_compat import is_ocr_budget_applicable, normalize_original_budget


def test_jpeg_area_not_applicable():
    assert is_ocr_budget_applicable("jpeg", "area_0.25") is False


def test_resize_area_applicable():
    assert is_ocr_budget_applicable("resize", "area_0.5") is True


def test_bops_patches_applicable():
    assert is_ocr_budget_applicable("bops", "patches_4") is True


def test_original_reference_normalized():
    assert normalize_original_budget("reference") == "area_1.0"
