"""Budget fairness enforcement for publishable preprocessing comparisons.

Every preprocessing method in BOPS must operate under a declared visual budget
(pixels, bytes, or patch count). This module validates that actual resource use
matches the target within tolerance and sets ``invalid_budget=True`` when it does
not. Rows marked invalid must be excluded from aggregate metrics and paper tables.

Tolerances (from the implementation plan):
    - Pixel (area): ±3% (``PIXEL_TOLERANCE``)
    - Byte (JPEG/WebP): valid if ``actual_bytes <= target_bytes``; report ``byte_utilization``
    - Patch count: exact match (zero tolerance)

Example::

    from src.utils.budget_check import check_pixel_budget, merge_budget_fields

    result = check_pixel_budget(actual_pixels=480000, target_pixels=500000)
    row = merge_budget_fields({"method": "resize"}, result)
    assert "invalid_budget" in row
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

PIXEL_TOLERANCE = 0.03
BYTE_UNDERUTILIZED_THRESHOLD = 0.70


@dataclass
class BudgetResult:
    """Outcome of a single budget compliance check.

    Attributes:
        budget_type: One of ``"pixel"``, ``"byte"``, or ``"patch"``.
        budget_target: Declared budget value.
        budget_actual: Measured value after preprocessing.
        invalid_budget: True if the result must be excluded from aggregates.
        tolerance: Relative tolerance used (0.0 for exact patch counts).
        byte_utilization: For byte budgets, ``actual / target`` (optional).
        underutilized_budget: For byte budgets, True if utilization < 0.70.
    """

    budget_type: str
    budget_target: float
    budget_actual: float
    invalid_budget: bool
    tolerance: float
    byte_utilization: float | None = None
    underutilized_budget: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a flat dict suitable for CSV/JSONL metadata rows."""
        return asdict(self)


def _within_tolerance(actual: float, target: float, tolerance: float) -> bool:
    """Return True if ``actual`` is within ``tolerance`` of ``target`` (relative)."""
    if target <= 0:
        return actual == target
    return abs(actual - target) / target <= tolerance


def check_pixel_budget(actual_pixels: int, target_pixels: int) -> BudgetResult:
    """Validate total pixel count against a target area budget.

    Args:
        actual_pixels: Width × height of the transformed image.
        target_pixels: Target pixel count (e.g. original_area × area_ratio).

    Returns:
        :class:`BudgetResult` with ``budget_type="pixel"``.
    """
    return BudgetResult(
        budget_type="pixel",
        budget_target=float(target_pixels),
        budget_actual=float(actual_pixels),
        invalid_budget=not _within_tolerance(actual_pixels, target_pixels, PIXEL_TOLERANCE),
        tolerance=PIXEL_TOLERANCE,
    )


def check_byte_budget(actual_bytes: int, target_bytes: int) -> BudgetResult:
    """Validate encoded file size against a byte budget.

    Valid when ``actual_bytes <= target_bytes``. Rows are invalid only when
    the encoded file exceeds the target. ``byte_utilization`` and
    ``underutilized_budget`` are reported for transparency (underutilized
    rows are not auto-excluded).

    Args:
        actual_bytes: Size of the compressed file on disk.
        target_bytes: Target size in bytes (e.g. 200 * 1024 for 200 KB).

    Returns:
        :class:`BudgetResult` with ``budget_type="byte"``.
    """
    utilization = (actual_bytes / target_bytes) if target_bytes > 0 else 0.0
    return BudgetResult(
        budget_type="byte",
        budget_target=float(target_bytes),
        budget_actual=float(actual_bytes),
        invalid_budget=actual_bytes > target_bytes,
        tolerance=0.0,
        byte_utilization=utilization,
        underutilized_budget=utilization < BYTE_UNDERUTILIZED_THRESHOLD,
    )


def check_patch_budget(actual_patches: int, target_patches: int) -> BudgetResult:
    """Validate patch count (must match exactly).

    Args:
        actual_patches: Number of high-resolution patches emitted.
        target_patches: Configured patch budget (e.g. 4 for overview + 4 patches).

    Returns:
        :class:`BudgetResult` with ``budget_type="patch"``.
    """
    return BudgetResult(
        budget_type="patch",
        budget_target=float(target_patches),
        budget_actual=float(actual_patches),
        invalid_budget=actual_patches != target_patches,
        tolerance=0.0,
    )


def merge_budget_fields(row: dict[str, Any], result: BudgetResult) -> dict[str, Any]:
    """Attach budget check fields to an existing metadata/result row in place.

    Args:
        row: Mutable dict (e.g. preprocessing metadata or eval result).
        result: Outcome from a ``check_*_budget`` function.

    Returns:
        The same ``row`` dict with budget fields merged in.
    """
    row.update(result.to_dict())
    return row
