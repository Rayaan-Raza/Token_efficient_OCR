"""Unit tests for budget fairness enforcement (:mod:`src.utils.budget_check`).

Validates pixel (±3%), byte (at-or-under-target), and exact patch-count checks.
"""

from src.utils.budget_check import (
    BYTE_UNDERUTILIZED_THRESHOLD,
    check_byte_budget,
    check_patch_budget,
    check_pixel_budget,
)


def test_pixel_budget_within_tolerance():
    """Pixel budget within ±3% should not be marked invalid."""
    r = check_pixel_budget(1000, 1000)
    assert r.invalid_budget is False


def test_pixel_budget_exceeds_tolerance():
    """Pixel budget beyond ±3% should be marked invalid."""
    r = check_pixel_budget(1100, 1000)
    assert r.invalid_budget is True


def test_byte_budget_exact_target():
    """Byte budget at target should be valid and fully utilized."""
    r = check_byte_budget(100_000, 100_000)
    assert r.invalid_budget is False
    assert r.byte_utilization == 1.0
    assert r.underutilized_budget is False


def test_byte_under_target_valid():
    """Files smaller than target are valid (not symmetric ±2%)."""
    r = check_byte_budget(80_000, 100_000)
    assert r.invalid_budget is False
    assert r.byte_utilization == 0.8
    assert r.underutilized_budget is False


def test_byte_underutilized_flag():
    """Low utilization is flagged but not invalid."""
    r = check_byte_budget(40_000, 100_000)
    assert r.invalid_budget is False
    assert r.byte_utilization == 0.4
    assert r.underutilized_budget is True
    assert BYTE_UNDERUTILIZED_THRESHOLD == 0.70


def test_byte_over_target_invalid():
    """Encoded size above target is invalid."""
    r = check_byte_budget(110_000, 100_000)
    assert r.invalid_budget is True


def test_patch_budget_exact():
    """Patch count must match target exactly; off-by-one is invalid."""
    r = check_patch_budget(4, 4)
    assert r.invalid_budget is False
    r2 = check_patch_budget(5, 4)
    assert r2.invalid_budget is True
