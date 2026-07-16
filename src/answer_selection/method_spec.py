"""Frozen RAVEN-Select production method specification.

The production method is a deterministic OCR-grounded answer-output selector.
Learned models and normalization variants are comparators / ablations only and
must not silently replace this rule after inspecting scaled or transfer results.
"""

from __future__ import annotations

from typing import Any

# Semantic version of the frozen production rule (bump only on intentional changes).
METHOD_ID = "raven_select"
METHOD_VERSION = "1.0.0"
METHOD_NAME = "RAVEN-Select"

PRODUCTION_READERS = ("resize", "bm25", "ler_bops")
# Exact length ties broken in this order.
TIE_BREAK_ORDER = ("resize", "bm25", "ler_bops")

NORMALIZATION = "conservative_lowercase_punct_ws"
GROUNDING = "page_or_route_patch_ocr_substring"
SELECTION = "shortest_ocr_grounded_else_shortest_nonempty"
COST_CALLS = 3

PRODUCTION_FLAGS = {
    "use_ocr": True,
    "use_length": True,
    "use_answer_type": False,
    "use_consensus": False,
}

LEAKAGE_EXCLUDED = (
    "gold_answer",
    "anls_to_gold",
    "em_to_gold",
    "answer_in_full_image_ocr",
    "answer_in_selected_patch_ocr",
    "oracle_route",
)


def method_stamp(**extra: Any) -> dict[str, Any]:
    """Compact provenance stamp for result artifacts."""
    stamp = {
        "method_id": METHOD_ID,
        "method_version": METHOD_VERSION,
        "method_name": METHOD_NAME,
        "readers": list(PRODUCTION_READERS),
        "tie_break_order": list(TIE_BREAK_ORDER),
        "normalization": NORMALIZATION,
        "grounding": GROUNDING,
        "selection": SELECTION,
        "vlm_calls": COST_CALLS,
        "production_flags": dict(PRODUCTION_FLAGS),
        "leakage_excluded": list(LEAKAGE_EXCLUDED),
        "role": "production",
    }
    stamp.update(extra)
    return stamp


def tie_rank(route: str) -> int:
    """Lower is better. Unknown routes sort after the production set."""
    try:
        return TIE_BREAK_ORDER.index(route)
    except ValueError:
        return len(TIE_BREAK_ORDER) + 1


__all__ = [
    "METHOD_ID",
    "METHOD_VERSION",
    "METHOD_NAME",
    "PRODUCTION_READERS",
    "TIE_BREAK_ORDER",
    "NORMALIZATION",
    "GROUNDING",
    "SELECTION",
    "COST_CALLS",
    "PRODUCTION_FLAGS",
    "LEAKAGE_EXCLUDED",
    "method_stamp",
    "tie_rank",
]
