"""Cost accounting helpers for QE-BOPS experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CostRecord:
    """Per-sample timing and budget accounting."""

    sample_id: str
    method: str
    candidate_gen_s: float = 0.0
    ocr_fullpage_s: float = 0.0
    ocr_patch_s: float = 0.0
    selection_s: float = 0.0
    vlm_s: float = 0.0
    num_candidates: int = 0
    num_patches_selected: int = 0
    num_vlm_images: int = 0
    median_pixels_selected: int = 0
    ocr_fullpage_cached: bool = False
    ocr_patch_cached: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def total_cold_s(self) -> float:
        return (
            self.candidate_gen_s
            + self.ocr_fullpage_s
            + self.ocr_patch_s
            + self.selection_s
            + self.vlm_s
        )

    @property
    def total_cached_s(self) -> float:
        return self.selection_s + self.vlm_s

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["total_cold_s"] = self.total_cold_s
        d["total_cached_s"] = self.total_cached_s
        return d


def aggregate_costs(records: list[CostRecord]) -> dict[str, float]:
    """Median aggregates for reporting."""
    if not records:
        return {}
    keys = [
        "candidate_gen_s",
        "ocr_fullpage_s",
        "ocr_patch_s",
        "selection_s",
        "vlm_s",
        "total_cold_s",
        "total_cached_s",
        "num_candidates",
        "num_vlm_images",
    ]

    def _median(vals: list[float]) -> float:
        s = sorted(vals)
        n = len(s)
        if n == 0:
            return 0.0
        mid = n // 2
        return float(s[mid]) if n % 2 else float((s[mid - 1] + s[mid]) / 2)

    out: dict[str, float] = {}
    for k in keys:
        if k.startswith("num_"):
            out[f"median_{k}"] = _median([float(getattr(r, k)) for r in records])
        else:
            out[f"median_{k}"] = _median([float(getattr(r, k) if hasattr(r, k) else r.as_dict()[k]) for r in records])
    return out
