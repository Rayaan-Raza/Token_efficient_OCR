"""Unit tests for budget fairness enforcement (:mod:`src.utils.budget_check`).

Validates pixel (±3%), byte (±2%), and exact patch-count checks used to flag
``invalid_budget`` rows before aggregating experiment metrics.
"""

from src.utils.budget_check import (
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


def test_byte_budget_within_tolerance():
    """Byte budget within ±2% should not be marked invalid."""
    r = check_byte_budget(100_000, 100_000)
    assert r.invalid_budget is False


def test_patch_budget_exact():
    """Patch count must match target exactly; off-by-one is invalid."""
    r = check_patch_budget(4, 4)
    assert r.invalid_budget is False
    r2 = check_patch_budget(5, 4)
    assert r2.invalid_budget is True
