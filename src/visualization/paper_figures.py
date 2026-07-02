"""Publication figures for the BOPS paper (PDF + PNG)."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from PIL import Image

from src.utils.experiment_io import filter_paper_dataframe
from src.utils.image_io import load_image
from src.utils.paths import data_path, repo_path


def _save(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _paper_ocr(df: pd.DataFrame) -> pd.DataFrame:
    return filter_paper_dataframe(df)


def _paper_vlm(df: pd.DataFrame) -> pd.DataFrame:
    return filter_paper_dataframe(df)


def plot_bops_pipeline(out_path: Path) -> None:
    """Schematic of overview + OCR-guided patch selection."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, color: str = "#E8F4FD") -> None:
        rect = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.2,
            edgecolor="#333333",
            facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, wrap=True)

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.2,
                color="#444444",
            )
        )

    box(0.2, 2.8, 1.4, 1.0, "Full-res\ndocument image", "#FFF3CD")
    box(2.2, 3.1, 1.6, 0.7, "Overview\n(low-res)", "#D1ECF1")
    box(2.2, 1.9, 1.6, 0.7, "Patch grid\n(256×256)", "#F8D7DA")

    box(4.4, 2.0, 1.8, 1.2, "OCR boxes +\nscore patches\n(text, conf,\nedge, entropy)", "#E2E3E5")
    box(6.8, 2.0, 1.5, 1.2, "NMS +\ntop-K select", "#E2E3E5")

    box(8.7, 3.0, 1.0, 0.8, "Overview", "#D1ECF1")
    box(8.7, 1.8, 1.0, 0.5, "Patch 1", "#CFE2FF")
    box(8.7, 1.1, 1.0, 0.5, "Patch K", "#CFE2FF")

    arrow(1.6, 3.3, 2.2, 3.45)
    arrow(1.6, 3.0, 2.2, 2.25)
    arrow(3.8, 2.25, 4.4, 2.6)
    arrow(6.2, 2.6, 6.8, 2.6)
    arrow(8.3, 2.9, 8.7, 3.4)
    arrow(8.3, 2.3, 8.7, 2.05)
    arrow(8.3, 2.1, 8.7, 1.35)

    ax.text(8.2, 0.35, "VLM / OCR downstream", ha="center", fontsize=10, style="italic")
    ax.set_title("BOPS: Budget-Aware OCR-Guided Overview-Plus-Patch Selection", fontsize=12, pad=8)
    _save(fig, out_path)


def plot_ocr_word_recall_budget(ocr_df: pd.DataFrame, out_path: Path) -> None:
    """Primary OCR figure: word recall vs budget (resize + BOPS)."""
    df = _paper_ocr(ocr_df)
    grouped = df.groupby(["method", "budget"], as_index=False)["word_recall"].mean()

    resize_order = ["area_1.0", "area_0.5", "area_0.25", "area_0.125"]
    bops_order = ["patches_2", "patches_4", "patches_8"]

    resize = grouped[grouped["method"] == "resize"].set_index("budget").reindex(resize_order)
    bops = grouped[grouped["method"] == "bops"].set_index("budget").reindex(bops_order)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    x_resize = np.arange(len(resize_order))
    x_bops = np.arange(len(bops_order)) + len(resize_order) + 0.8

    ax.plot(
        x_resize,
        resize["word_recall"],
        marker="o",
        linewidth=2,
        color="#1f77b4",
        label="resize (area ratio)",
    )
    ax.plot(
        x_bops,
        bops["word_recall"],
        marker="s",
        linewidth=2,
        color="#d62728",
        label="BOPS (patch count)",
    )

    for xi, val in zip(x_resize, resize["word_recall"]):
        ax.annotate(f"{val:.3f}", (xi, val), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    for xi, val in zip(x_bops, bops["word_recall"]):
        ax.annotate(f"{val:.3f}", (xi, val), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    # Key comparison bracket
    y_resize = float(resize.loc["area_0.25", "word_recall"])
    y_bops = float(bops.loc["patches_8", "word_recall"])
    x_r = x_resize[resize_order.index("area_0.25")]
    x_b = x_bops[bops_order.index("patches_8")]
    y_br = max(y_resize, y_bops) + 0.04
    ax.plot([x_r, x_b], [y_br, y_br], color="#333333", linewidth=1)
    ax.plot([x_r, x_r], [y_resize + 0.01, y_br], color="#333333", linewidth=1)
    ax.plot([x_b, x_b], [y_bops + 0.01, y_br], color="#333333", linewidth=1)
    ax.text(
        (x_r + x_b) / 2,
        y_br + 0.01,
        "Δ = +0.077\n95% CI [0.052, 0.102]",
        ha="center",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    tick_pos = list(x_resize) + list(x_bops)
    tick_labels = [b.replace("area_", "area=") for b in resize_order] + [
        b.replace("patches_", "K=") for b in bops_order
    ]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=0)
    ax.axvline(len(resize_order) + 0.4, color="#cccccc", linestyle="--", linewidth=1)
    ax.text(len(resize_order) / 2 - 0.5, -0.08, "Resize budgets", transform=ax.get_xaxis_transform(), ha="center", fontsize=9)
    ax.text(
        len(resize_order) + 0.8 + len(bops_order) / 2 - 0.5,
        -0.08,
        "BOPS patch budgets",
        transform=ax.get_xaxis_transform(),
        ha="center",
        fontsize=9,
    )

    ax.set_ylabel("Word recall")
    ax.set_ylim(0, max(grouped["word_recall"].max() + 0.12, 0.35))
    ax.set_title("OCR word recall vs visual budget (TextOCR pilot, n=200)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_path)


def plot_vlm_anls_methods(vlm_df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart of VLM ANLS and EM by method."""
    df = _paper_vlm(vlm_df)
    order = ["resize", "uniform", "bops", "bops_qa", "overview_only", "random"]
    grouped = df.groupby("method", as_index=False).agg(anls=("anls", "mean"), em=("exact_match", "mean"))
    present = [m for m in order if m in set(grouped["method"])]
    grouped = grouped.set_index("method").reindex(present).reset_index()

    x = np.arange(len(present))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, grouped["anls"], width, label="ANLS", color="#4C78A8")
    bars2 = ax.bar(x + width / 2, grouped["em"], width, label="Exact Match", color="#F58518")

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", (bar.get_x() + bar.get_width() / 2, h), ha="center", va="bottom", fontsize=8)

    labels = [m.replace("_", "\n") if m == "overview_only" else m.replace("_", " ") for m in present]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.95)
    ax.set_title("DocVQA pilot (n=100, K=2): VLM performance by preprocessing method")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_path)


def plot_answer_coverage_diagnostics(vlm_df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart: patch OCR answer coverage by VLM preprocessing method."""
    df = _paper_vlm(vlm_df)

    def _rate(series: pd.Series) -> float:
        vals = series
        if vals.dtype == object:
            vals = vals.map(lambda v: str(v).lower() in ("true", "1", "1.0"))
        return float(vals.fillna(False).astype(bool).mean())

    methods = ["bops", "bops_qa", "random", "uniform", "overview_only"]
    present = [m for m in methods if m in set(df["method"])]
    if not present:
        present = sorted(df["method"].unique())

    full_rates = []
    patch_rates = []
    for m in present:
        sub = df[df["method"] == m]
        full_rates.append(_rate(sub["answer_in_full_image_ocr"]) * 100)
        patch_rates.append(_rate(sub["answer_in_selected_patch_ocr"]) * 100)

    x = np.arange(len(present))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - width / 2, full_rates, width, label="Full-image OCR", color="#59A14F")
    b2 = ax.bar(x + width / 2, patch_rates, width, label="Selected-patch OCR", color="#E15759")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.0f}%", (bar.get_x() + bar.get_width() / 2, h), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(present, rotation=15, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("% samples with answer in OCR text")
    ax.set_title("Answer coverage by patch selector (DocVQA pilot, n=100, K=2)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_path)


def _parse_answers(raw: str) -> list[str]:
    try:
        val = json.loads(raw)
    except json.JSONDecodeError:
        try:
            val = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            val = [raw]
    if isinstance(val, str):
        return [val]
    return list(val)


def _normalize_match_text(text: str) -> str:
    return " ".join(text.lower().split())


def _answer_substring_in_text(answers: list[str], text: str) -> bool:
    norm = _normalize_match_text(text)
    for ans in answers:
        a = _normalize_match_text(ans)
        if a and a in norm:
            return True
    return False


def _box_matches_answer(answers: list[str], box_text: str) -> bool:
    """Return True if OCR box text overlaps any reference answer."""
    norm_box = _normalize_match_text(box_text)
    if not norm_box:
        return False
    for ans in answers:
        norm_ans = _normalize_match_text(ans)
        if not norm_ans:
            continue
        if norm_ans in norm_box or norm_box in norm_ans:
            return True
        for tok in norm_ans.split():
            if len(tok) >= 3 and tok in norm_box:
                return True
    return False


def _answer_boxes(boxes: list[dict[str, Any]], answers: list[str]) -> list[dict[str, Any]]:
    return [b for b in boxes if _box_matches_answer(answers, b.get("text", ""))]


def _draw_patch_rect(ax, x: float, y: float, w: float, h: float, *, color: str, linestyle: str) -> None:
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            linewidth=2.5,
            edgecolor=color,
            facecolor="none",
            linestyle=linestyle,
        )
    )


def _draw_answer_box(ax, box: list, *, facecolor: str = "#59A14F", edgecolor: str = "#1B5E20") -> None:
    pts = [(float(p[0]), float(p[1])) for p in box]
    ax.add_patch(
        Polygon(
            pts,
            closed=True,
            fill=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=2.5,
            alpha=0.35,
        )
    )


def _collect_failure_examples(
    vlm_df: pd.DataFrame,
    manifest_path: Path,
    *,
    max_examples: int,
    skip: int = 0,
) -> list[tuple[int, pd.Series, list[dict[str, Any]]]]:
    """Rank BOPS failures where full-image OCR has the answer but selected patches do not."""
    from src.ocr.run_ocr import run_ocr_with_boxes
    import tempfile

    df = _paper_vlm(vlm_df)
    bops = df[df["method"] == "bops"].copy()
    if len(bops) == 0:
        return []

    def _bool_col(series: pd.Series) -> pd.Series:
        return series.map(lambda v: str(v).lower() in ("true", "1", "1.0"))

    candidates = bops[
        _bool_col(bops["answer_in_full_image_ocr"])
        & ~_bool_col(bops["answer_in_selected_patch_ocr"])
    ].copy()
    candidates = candidates[candidates["exact_match"] < 1.0]
    if len(candidates) == 0:
        return []

    candidates["q_len"] = candidates["question"].astype(str).str.len()
    candidates = candidates.sort_values(["q_len", "mean_patch_score"], ascending=[True, False])

    manifest = {r["image_id"]: r for r in _iter_manifest_records(manifest_path)}

    shortlist = candidates.head(max((skip + max_examples) * 4, 12))
    ranked: list[tuple[int, pd.Series, list[dict[str, Any]]]] = []
    for _, row in shortlist.iterrows():
        record = manifest.get(row["image_id"])
        if not record:
            continue
        image = load_image(repo_path(record["image_path"]))
        answers = _parse_answers(str(row["ground_truth_answer"]))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            boxes = run_ocr_with_boxes(tmp.name)
        n_hits = len(_answer_boxes(boxes, answers))
        ranked.append((n_hits, row, boxes))
    ranked.sort(key=lambda t: (-t[0], t[1]["q_len"]))
    return ranked[skip : skip + max_examples]


def _format_failure_caption(row: pd.Series, answers: list[str]) -> str:
    q = str(row["question"])
    if len(q) > 120:
        q = q[:117] + "..."
    pred = str(row["parsed_prediction"])
    if len(pred) > 60:
        pred = pred[:57] + "..."
    return (
        f"{row['image_id']}  |  Q: {q}\n"
        f"GT: {answers[0]!r}  |  pred: {pred!r}  |  "
        f"answer in full OCR: yes  |  in selected patches: no"
    )


def _render_failure_row(
    axes_row: np.ndarray,
    row: pd.Series,
    boxes: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    panel_title_size: int,
    caption_size: int,
    show_caption: bool = True,
) -> None:
    image_id = row["image_id"]
    record = manifest.get(image_id)
    if not record:
        return
    image = load_image(repo_path(record["image_path"]))
    answers = _parse_answers(str(row["ground_truth_answer"]))
    answer_hits = _answer_boxes(boxes, answers)

    ax_img, ax_annot = axes_row
    ax_img.imshow(image)
    ax_img.axis("off")
    coords = json.loads(row["selected_patch_coords"]) if pd.notna(row["selected_patch_coords"]) else []
    for x, y, w, h in coords:
        _draw_patch_rect(ax_img, x, y, w, h, color="#E15759", linestyle="--")
    ax_img.set_title("Selected BOPS patches (red)", fontsize=panel_title_size, pad=8)

    ax_annot.imshow(image)
    ax_annot.axis("off")
    for b in answer_hits:
        _draw_answer_box(ax_annot, b["box"])
    if not answer_hits:
        ax_annot.text(
            0.5,
            0.02,
            "No single OCR box matched answer text",
            transform=ax_annot.transAxes,
            ha="center",
            fontsize=caption_size,
            color="#B71C1C",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )
    ax_annot.set_title("Answer OCR boxes (green)", fontsize=panel_title_size, pad=8)

    if show_caption:
        caption = _format_failure_caption(row, answers)
        ax_img.text(
            0.5,
            -0.06,
            caption,
            transform=ax_img.transAxes,
            ha="center",
            va="top",
            fontsize=caption_size,
            wrap=True,
        )


def plot_failure_panel(
    vlm_df: pd.DataFrame,
    manifest_path: Path,
    out_path: Path,
    *,
    max_examples: int = 2,
    skip: int = 0,
    layout: str = "grid",
    figsize_scale: float = 1.0,
    title_suffix: str = "",
) -> None:
    """Visual panel of BOPS patch-selection failures."""
    manifest = {r["image_id"]: r for r in _iter_manifest_records(manifest_path)}
    panel_rows = _collect_failure_examples(
        vlm_df,
        manifest_path,
        max_examples=max_examples,
        skip=skip,
    )
    if not panel_rows:
        return

    red_patch = mpatches.Patch(edgecolor="#E15759", facecolor="none", linewidth=2, label="Selected patch")
    green_box = mpatches.Patch(facecolor="#59A14F", edgecolor="#1B5E20", alpha=0.35, label="OCR box with answer")

    if layout == "hero":
        fig = plt.figure(figsize=(13.5 * figsize_scale, 7.0 * figsize_scale))
        gs = fig.add_gridspec(2, 2, height_ratios=[5.8, 1.0], hspace=0.32, wspace=0.06)
        ax_img = fig.add_subplot(gs[0, 0])
        ax_annot = fig.add_subplot(gs[0, 1])
        _, row, boxes = panel_rows[0]
        answers = _parse_answers(str(row["ground_truth_answer"]))
        image_id = row["image_id"]
        record = manifest.get(image_id)
        if not record:
            return
        image = load_image(repo_path(record["image_path"]))
        answer_hits = _answer_boxes(boxes, answers)

        ax_img.imshow(image)
        ax_img.axis("off")
        coords = json.loads(row["selected_patch_coords"]) if pd.notna(row["selected_patch_coords"]) else []
        for x, y, w, h in coords:
            _draw_patch_rect(ax_img, x, y, w, h, color="#E15759", linestyle="--")
        ax_img.set_title("Selected BOPS patches (red)", fontsize=13, pad=10)

        ax_annot.imshow(image)
        ax_annot.axis("off")
        for b in answer_hits:
            _draw_answer_box(ax_annot, b["box"])
        ax_annot.set_title("Answer OCR boxes (green)", fontsize=13, pad=10)

        caption_ax = fig.add_subplot(gs[1, :])
        caption_ax.axis("off")
        caption_ax.text(
            0.5,
            0.72,
            _format_failure_caption(row, answers),
            ha="center",
            va="center",
            fontsize=11,
            linespacing=1.35,
            bbox=dict(boxstyle="round", facecolor="#F8F9FA", edgecolor="#CCCCCC", pad=0.8),
        )
        caption_ax.legend(
            handles=[red_patch, green_box],
            loc="lower center",
            ncol=2,
            fontsize=11,
            frameon=False,
        )
        fig.suptitle(
            f"BOPS patch-selection failure (DocVQA pilot){title_suffix}",
            fontsize=14,
            y=0.98,
        )
        fig.subplots_adjust(bottom=0.10, top=0.90, left=0.03, right=0.97)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path.with_suffix(".pdf"))
        fig.savefig(out_path.with_suffix(".png"), dpi=200)
        plt.close(fig)
        return

    n = len(panel_rows)
    row_h = 4.8 * figsize_scale if layout == "appendix" else 3.2 * figsize_scale
    fig, axes = plt.subplots(n, 2, figsize=(12 * figsize_scale, row_h * n))
    if n == 1:
        axes = np.array([axes])

    title_size = 11 if layout == "appendix" else 9
    caption_size = 10 if layout == "appendix" else 8
    for row_idx, (_, row, boxes) in enumerate(panel_rows):
        _render_failure_row(
            axes[row_idx],
            row,
            boxes,
            manifest,
            panel_title_size=title_size,
            caption_size=caption_size,
            show_caption=False,
        )
        q_short = str(row["question"])
        if len(q_short) > 80:
            q_short = q_short[:77] + "..."
        axes[row_idx, 0].set_title(
            f"{row['image_id']}  |  Q: {q_short}",
            fontsize=title_size,
            pad=10,
        )

    fig.legend(handles=[red_patch, green_box], loc="lower center", ncol=2, fontsize=10)
    fig.suptitle(
        f"BOPS patch-selection failures (DocVQA pilot){title_suffix}",
        fontsize=12,
        y=0.995,
    )
    fig.subplots_adjust(bottom=0.05, top=0.93, hspace=0.42, wspace=0.08)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def _iter_manifest_records(path: Path):
    from src.data.dataset_loader import iter_manifest

    yield from iter_manifest(path)


def plot_runtime_comparison(ocr_df: pd.DataFrame, vlm_df: pd.DataFrame, out_path: Path) -> None:
    """Runtime bar chart for OCR and VLM methods."""
    ocr = _paper_ocr(ocr_df)
    vlm = _paper_vlm(vlm_df)

    ocr_rt = ocr.groupby("method")["runtime_sec"].median().reindex(["resize", "jpeg", "webp", "bops"])
    vlm_rt = vlm.groupby("method")["runtime_sec"].median().reindex(
        ["resize", "uniform", "random", "overview_only", "bops"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].bar(ocr_rt.index.astype(str), ocr_rt.values, color="#72B7B2")
    axes[0].set_title("OCR median runtime / image-method (s)")
    axes[0].set_ylabel("Seconds (median)")
    axes[0].tick_params(axis="x", rotation=25)
    for i, v in enumerate(ocr_rt.values):
        axes[0].annotate(f"{v:.2f}", (i, v), ha="center", va="bottom", fontsize=8)

    axes[1].bar(vlm_rt.index.astype(str), vlm_rt.values, color="#B279A2")
    axes[1].set_title("VLM median runtime / sample (s)")
    axes[1].set_ylabel("Seconds (median)")
    axes[1].tick_params(axis="x", rotation=25)
    for i, v in enumerate(vlm_rt.values):
        axes[1].annotate(f"{v:.1f}", (i, v), ha="center", va="bottom", fontsize=8)

    fig.suptitle("Runtime cost: BOPS is slower; resize is fast and strong on VLM", fontsize=11)
    fig.tight_layout()
    _save(fig, out_path)


def write_runtime_table(ocr_df: pd.DataFrame, vlm_df: pd.DataFrame, out_path: Path) -> None:
    """Write runtime summary CSV for the paper."""
    ocr = _paper_ocr(ocr_df)
    vlm = _paper_vlm(vlm_df)

    rows: list[dict[str, Any]] = []
    for track, df in ("ocr", ocr), ("vlm", vlm):
        for method, g in df.groupby("method"):
            rows.append(
                {
                    "track": track,
                    "method": method,
                    "n": len(g),
                    "runtime_mean_sec": round(float(g["runtime_sec"].mean()), 3),
                    "runtime_median_sec": round(float(g["runtime_sec"].median()), 3),
                    "runtime_p90_sec": round(float(g["runtime_sec"].quantile(0.9)), 3),
                }
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["track", "method"]).to_csv(out_path, index=False)
