#!/usr/bin/env python3
"""Build budget/cost summary table for the paper."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.utils.experiment_io import filter_paper_dataframe
from src.utils.paths import outputs_path, repo_path


def _vlm_images_sent(method: str, num_patches: int) -> int:
    if method == "resize":
        return 1
    if method == "overview_only":
        return 1
    return 1 + int(num_patches)  # overview + K patches


def main() -> None:
    ocr_candidates = sorted(
        outputs_path("metrics").glob("ocr_metrics*pilot*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    vlm_path = outputs_path("metrics", "vlm_metrics_merged.csv")
    if not ocr_candidates or not vlm_path.exists():
        raise FileNotFoundError("Need pilot OCR and merged VLM CSVs")

    ocr = filter_paper_dataframe(pd.read_csv(ocr_candidates[0]))
    vlm = filter_paper_dataframe(pd.read_csv(vlm_path))

    rows: list[dict] = []

    # OCR methods — one row per method×budget aggregate
    for (method, budget), g in ocr.groupby(["method", "budget"]):
        row = {
            "track": "ocr",
            "method": method,
            "budget": budget,
            "budget_type": g["budget_type"].iloc[0] if "budget_type" in g else "",
            "actual_pixels_median": g["budget_actual"].median() if "budget_actual" in g else None,
            "actual_bytes_median": None,
            "vlm_images_sent": None,
            "num_patches": None,
            "median_ocr_runtime_sec": round(float(g["runtime_sec"].median()), 3),
            "median_vlm_runtime_sec": None,
            "n": len(g),
        }
        if method in ("jpeg", "webp") and "byte_utilization" in g.columns:
            util = g["byte_utilization"].median()
            if pd.notna(util) and "budget_target" in g.columns:
                row["actual_bytes_median"] = round(float(g["budget_target"].iloc[0] * util), 0)
        if method == "bops" and "total_bops_pixels" in g.columns:
            row["actual_pixels_median"] = round(float(g["total_bops_pixels"].median()), 0)
        rows.append(row)

    # VLM methods — one row per method
    for method, g in vlm.groupby("method"):
        k = int(g["num_patches"].iloc[0]) if "num_patches" in g else 2
        rows.append({
            "track": "vlm",
            "method": method,
            "budget": f"patches_{k}" if method != "resize" else "area_0.25",
            "budget_type": "patch" if method != "resize" else "pixel",
            "actual_pixels_median": None,
            "actual_bytes_median": None,
            "vlm_images_sent": _vlm_images_sent(method, k),
            "num_patches": k if method not in ("resize",) else 0,
            "median_ocr_runtime_sec": None,
            "median_vlm_runtime_sec": round(float(g["runtime_sec"].median()), 3),
            "n": len(g),
        })

    out_dirs = [
        repo_path("paper", "tables"),
        repo_path("paper", "latex_v2", "tables"),
    ]
    df = pd.DataFrame(rows)
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        path = d / "table_budget_cost.csv"
        df.to_csv(path, index=False)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
