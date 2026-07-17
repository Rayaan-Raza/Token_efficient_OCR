"""Build leakage-safe prediction-in-OCR presence cache for RAVEN-Select.

Uses cached full-page OCR boxes and optional patch OCR text. Checks whether the
*prediction* (not gold) appears in OCR text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.routing import normalize_pred
from src.routing.train import METHOD_FILES, load_methods
from src.utils.ocr_cache import DEFAULT_OCR_ENGINE, load_cached_ocr_boxes, load_cached_patch_ocr
from src.utils.paths import outputs_path

DEFAULT_METHODS = ["resize", "bm25", "ler_bops"]
# Map route name used in selection → VLM method name used in patch OCR cache keys
_ROUTE_TO_VLM_METHOD = {
    "resize": "resize",
    "bm25": "bm25_only",
    "ler_bops": "learned_lgbm_strict",
}


def _boxes_to_text(boxes: list[dict] | None) -> str:
    if not boxes:
        return ""
    return " ".join(str(b.get("text", "")) for b in boxes)


def build_ocr_presence_cache(
    n: int,
    methods: Sequence[str] | None = None,
    *,
    metrics_tag: str = "",
    num_patches: int = 2,
    out_path: Path | None = None,
    ocr_engine: str | None = None,
    dataset: str = "docvqa",
) -> Path:
    """Write model-tagged OCR-presence rows for each (image_id, route)."""
    methods = list(methods or DEFAULT_METHODS)
    data = load_methods(n, methods, metrics_tag=metrics_tag, dataset=dataset)
    engine = ocr_engine or DEFAULT_OCR_ENGINE
    rows = []
    missing_full = 0
    for iid in data.index:
        boxes = load_cached_ocr_boxes(str(iid), engine=engine)
        full_text = _boxes_to_text(boxes)
        if not full_text:
            missing_full += 1
        for m in methods:
            pred = str(data.loc[iid, f"pred__{m}"])
            npred = normalize_pred(pred)
            vlm_m = _ROUTE_TO_VLM_METHOD.get(m, m)
            patch_payload = load_cached_patch_ocr(str(iid), vlm_m, num_patches)
            patch_texts = list((patch_payload or {}).get("patch_texts", []) or [])
            patch_text = " ".join(patch_texts)
            full_n = normalize_pred(full_text)
            patch_n = normalize_pred(patch_text)
            rows.append({
                "image_id": iid,
                "route": m,
                "ocr_engine": engine,
                "prediction": pred,
                "full_ocr_text": full_text[:5000],  # cap size
                "patch_ocr_text": patch_text[:5000],
                "pred_in_full_ocr": bool(npred and npred in full_n),
                "pred_in_patch_ocr": bool(npred and npred in patch_n),
                "has_full_ocr": bool(full_text),
                "has_patch_ocr": bool(patch_text),
            })
    suffix = f"_{metrics_tag}" if metrics_tag else ""
    engine_suffix = f"_{engine}" if engine != DEFAULT_OCR_ENGINE else ""
    ds = "" if dataset == "docvqa" else f"_{dataset}"
    out = out_path or outputs_path(
        "metrics",
        f"raven_select_ocr_presence{ds}_n{n}{engine_suffix}{suffix}.parquet",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    meta = {
        "n": n,
        "metrics_tag": metrics_tag,
        "ocr_engine": engine,
        "rows": len(df),
        "missing_full_ocr_images": missing_full,
        "frac_pred_in_full": float(df["pred_in_full_ocr"].mean()),
        "frac_pred_in_patch": float(df["pred_in_patch_ocr"].mean()),
        "path": str(out),
    }
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out
