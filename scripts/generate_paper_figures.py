#!/usr/bin/env python3
"""Generate publication figures for paper/draft.tex.

Writes PDF + PNG to ``paper/figures/`` and runtime table to ``paper/tables/``.

Run::

    python scripts/generate_paper_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.utils.paths import data_path, outputs_path, repo_path
from src.visualization.paper_figures import (
    plot_answer_coverage_diagnostics,
    plot_bops_pipeline,
    plot_failure_panel,
    plot_ocr_word_recall_budget,
    plot_runtime_comparison,
    plot_vlm_anls_methods,
    write_runtime_table,
)


def _default_ocr_csv() -> Path:
    metrics_dir = outputs_path("metrics")
    candidates = sorted(
        metrics_dir.glob("ocr_metrics*pilot*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    return metrics_dir / "ocr_metrics.csv"


def _default_vlm_csv() -> Path:
    merged = outputs_path("metrics", "vlm_metrics_merged.csv")
    return merged if merged.exists() else outputs_path("metrics", "vlm_metrics.csv")


def _default_manifest() -> Path:
    for name in ("docvqa_pilot.jsonl", "docvqa_debug.jsonl"):
        p = data_path("manifests", name)
        if p.exists():
            return p
    return data_path("manifests", "docvqa_pilot.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures (PDF + PNG).")
    parser.add_argument("--ocr-csv", type=Path, default=None)
    parser.add_argument("--vlm-csv", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=repo_path("paper", "figures"))
    parser.add_argument("--skip-failure-panel", action="store_true", help="Skip OCR-heavy failure panel")
    args = parser.parse_args()

    ocr_csv = args.ocr_csv or _default_ocr_csv()
    vlm_csv = args.vlm_csv or _default_vlm_csv()
    manifest = args.manifest or _default_manifest()
    out_dir = args.out_dir

    if not ocr_csv.exists():
        print(f"Missing OCR CSV: {ocr_csv}")
        sys.exit(1)
    if not vlm_csv.exists():
        print(f"Missing VLM CSV: {vlm_csv}")
        sys.exit(1)

    ocr_df = pd.read_csv(ocr_csv)
    vlm_df = pd.read_csv(vlm_csv)

    figures = [
        ("bops_pipeline", lambda: plot_bops_pipeline(out_dir / "bops_pipeline")),
        ("ocr_word_recall_budget", lambda: plot_ocr_word_recall_budget(ocr_df, out_dir / "ocr_word_recall_budget")),
        ("vlm_anls_methods", lambda: plot_vlm_anls_methods(vlm_df, out_dir / "vlm_anls_methods")),
        ("answer_coverage_diagnostics", lambda: plot_answer_coverage_diagnostics(vlm_df, out_dir / "answer_coverage_diagnostics")),
        ("runtime_comparison", lambda: plot_runtime_comparison(ocr_df, vlm_df, out_dir / "runtime_comparison")),
    ]
    if not args.skip_failure_panel:
        figures.append(
            ("failure_panel", lambda: plot_failure_panel(
                vlm_df, manifest, out_dir / "failure_panel",
                max_examples=1, layout="hero", figsize_scale=1.0,
            )),
        )
        figures.append(
            ("failure_panel_appendix", lambda: plot_failure_panel(
                vlm_df, manifest, out_dir / "failure_panel_appendix",
                max_examples=2, skip=1, layout="appendix", figsize_scale=1.35,
                title_suffix=" (supplementary)",
            )),
        )

    for name, fn in figures:
        print(f"Generating {name} ...")
        fn()
        print(f"  -> {out_dir / name}.pdf")

    latex_fig_dir = repo_path("paper", "latex", "figures")
    latex_v2_fig_dir = repo_path("paper", "latex_v2", "figures")
    for fig_dir in (latex_fig_dir, latex_v2_fig_dir):
        fig_dir.mkdir(parents=True, exist_ok=True)
        for pdf in out_dir.glob("*.pdf"):
            dest = fig_dir / pdf.name
            dest.write_bytes(pdf.read_bytes())
            print(f"  synced -> {dest}")

    runtime_table = repo_path("paper", "tables", "table_runtime.csv")
    write_runtime_table(ocr_df, vlm_df, runtime_table)
    print(f"Wrote {runtime_table}")


if __name__ == "__main__":
    main()
