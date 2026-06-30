"""Matplotlib figures for paper and pilot experiments.

Regenerates budget degradation curves from CSV metric files (no screenshot plots).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_cer_vs_budget(csv_path: Path, out_path: Path, budget_col: str = "budget") -> None:
    """Plot mean CER vs budget for each preprocessing method.

    Excludes rows with ``invalid_budget=True`` and missing CER values.

    Args:
        csv_path: Input ``ocr_metrics.csv`` from OCR evaluation.
        out_path: Output PNG path.
        budget_col: Column name for x-axis budget labels.
    """
    df = pd.read_csv(csv_path)
    if "cer" not in df.columns:
        print(f"Skipping plot: no cer column in {csv_path}")
        return
    valid = df.copy()
    if "invalid_budget" in valid.columns:
        valid = valid[valid["invalid_budget"].fillna(False).astype(bool) == False]
    valid = valid.dropna(subset=["cer"])
    grouped = valid.groupby(["method", budget_col])["cer"].mean().reset_index()
    plt.figure(figsize=(8, 5))
    for method, g in grouped.groupby("method"):
        plt.plot(g[budget_col], g["cer"], marker="o", label=method)
    plt.xlabel(budget_col)
    plt.ylabel("CER")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
