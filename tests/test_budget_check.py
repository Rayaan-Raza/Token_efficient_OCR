from src.utils.budget_check import (
    check_byte_budget,
    check_patch_budget,
    check_pixel_budget,
)


def test_pixel_budget_within_tolerance():
    r = check_pixel_budget(1000, 1000)
    assert r.invalid_budget is False


def test_pixel_budget_exceeds_tolerance():
    r = check_pixel_budget(1100, 1000)
    assert r.invalid_budget is True


def test_byte_budget_within_tolerance():
    r = check_byte_budget(100_000, 100_000)
    assert r.invalid_budget is False


def test_patch_budget_exact():
    r = check_patch_budget(4, 4)
    assert r.invalid_budget is False
    r2 = check_patch_budget(5, 4)
    assert r2.invalid_budget is True
