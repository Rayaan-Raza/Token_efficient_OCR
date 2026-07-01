"""Method × budget compatibility for fair experiment comparisons.

Defines which budget tokens apply to each preprocessing method. Incompatible
combinations must be flagged ``not_applicable=true`` (not ``invalid_budget``).

``invalid_budget`` is reserved for cases where a method attempted to hit a
valid budget but missed tolerance.
"""

from __future__ import annotations

# OCR evaluation methods and their valid budget tokens
OCR_METHOD_BUDGETS: dict[str, frozenset[str]] = {
    "original": frozenset({"reference", "area_1.0"}),
    "resize": frozenset({"area_1.0", "area_0.5", "area_0.25", "area_0.125"}),
    "jpeg": frozenset({"kb_500", "kb_200", "kb_100", "kb_50"}),
    "webp": frozenset({"kb_500", "kb_200", "kb_100", "kb_50"}),
    "bops": frozenset({"patches_2", "patches_4", "patches_8", "patches_12"}),
}

VALID_EXPERIMENT_STAGES = frozenset({"debug", "sanity", "pilot", "paper"})


def is_ocr_budget_applicable(method: str, budget: str) -> bool:
    """Return True if ``budget`` is a valid axis for ``method`` in OCR eval.

    Args:
        method: Preprocessing method name (e.g. ``resize``, ``jpeg``).
        budget: Budget token (e.g. ``area_0.5``, ``kb_200``, ``patches_4``).

    Returns:
        True when the pair should be evaluated; False → ``not_applicable``.
    """
    allowed = OCR_METHOD_BUDGETS.get(method)
    if allowed is None:
        return False
    return budget in allowed


def normalize_original_budget(budget: str) -> str:
    """Map ``reference`` to ``area_1.0`` for consistent reporting."""
    return "area_1.0" if budget == "reference" else budget
