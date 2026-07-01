"""Tests for BOPS patch OCR text merging."""

from src.ocr.merge_patch_ocr import merge_patch_ocr


def test_merge_overview_and_patches_dedup_tokens():
    overview = "hello world"
    patch1 = "world foo"
    merged = merge_patch_ocr([overview, patch1])
    tokens = merged.split()
    assert "hello" in tokens
    assert "world" in tokens
    assert "foo" in tokens
    assert tokens.count("world") == 1


def test_merge_empty_segments():
    assert merge_patch_ocr(["", "  ", "bar"]) == "bar"
