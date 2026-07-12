"""Tests for G1 candidate-pool coverage metrics."""

from PIL import Image

from src.preprocessing.candidate_pool import compute_pool_coverage_stats
from src.preprocessing.patch_grid import Patch


def test_unique_image_area_coverage_is_clipped_union():
    image = Image.new("RGB", (100, 100), color="white")
    full = Patch(0, 0, 100, 100, 0)
    stats = compute_pool_coverage_stats(image, [full], [])
    assert stats["unique_image_area_coverage"] == 1.0


def test_candidate_area_ratio_counts_overlaps():
    image = Image.new("RGB", (100, 100), color="white")
    p1 = Patch(0, 0, 100, 100, 0)
    p2 = Patch(0, 0, 100, 100, 1)
    stats = compute_pool_coverage_stats(image, [p1, p2], [])
    assert stats["unique_image_area_coverage"] == 1.0
    assert stats["candidate_area_ratio"] == 2.0


def test_unique_coverage_never_exceeds_one_when_dims_not_divisible_by_step():
    image = Image.new("RGB", (101, 103), color="white")
    full = Patch(0, 0, 101, 103, 0)
    stats = compute_pool_coverage_stats(image, [full], [])
    assert stats["unique_image_area_coverage"] <= 1.0
    assert stats["candidate_area_ratio"] > 0.99
