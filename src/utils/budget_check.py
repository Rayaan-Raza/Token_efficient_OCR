"""Budget fairness enforcement for publishable comparisons."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

PIXEL_TOLERANCE = 0.03
BYTE_TOLERANCE = 0.02


@dataclass
class BudgetResult:
    budget_type: str
    budget_target: float
    budget_actual: float
    invalid_budget: bool
    tolerance: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _within_tolerance(actual: float, target: float, tolerance: float) -> bool:
    if target <= 0:
        return actual == target
    return abs(actual - target) / target <= tolerance


def check_pixel_budget(actual_pixels: int, target_pixels: int) -> BudgetResult:
    return BudgetResult(
        budget_type="pixel",
        budget_target=float(target_pixels),
        budget_actual=float(actual_pixels),
        invalid_budget=not _within_tolerance(actual_pixels, target_pixels, PIXEL_TOLERANCE),
        tolerance=PIXEL_TOLERANCE,
    )


def check_byte_budget(actual_bytes: int, target_bytes: int) -> BudgetResult:
    return BudgetResult(
        budget_type="byte",
        budget_target=float(target_bytes),
        budget_actual=float(actual_bytes),
        invalid_budget=not _within_tolerance(actual_bytes, target_bytes, BYTE_TOLERANCE),
        tolerance=BYTE_TOLERANCE,
    )


def check_patch_budget(actual_patches: int, target_patches: int) -> BudgetResult:
    return BudgetResult(
        budget_type="patch",
        budget_target=float(target_patches),
        budget_actual=float(actual_patches),
        invalid_budget=actual_patches != target_patches,
        tolerance=0.0,
    )


def merge_budget_fields(row: dict[str, Any], result: BudgetResult) -> dict[str, Any]:
    row.update(result.to_dict())
    return row
