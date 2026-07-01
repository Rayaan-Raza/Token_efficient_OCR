"""Shared helpers for experiment runs: run IDs, CSV paths, and paper filters."""

from __future__ import annotations

import csv
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.paths import outputs_path

PAPER_STAGES = frozenset({"pilot", "paper"})


def new_run_id() -> str:
    """Generate a short unique run identifier."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]


def iso_timestamp() -> str:
    """Return current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def manifest_stem(manifest_path: str | Path) -> str:
    """Extract manifest filename without ``.jsonl`` extension."""
    return Path(manifest_path).stem


def vlm_patch_suffix(method: str, num_patches: int) -> str:
    """Build patch-count suffix for VLM metric filenames.

    Args:
        method: VLM eval method.
        num_patches: Configured patch budget.

    Returns:
        ``single`` for resize, ``k0`` for overview_only, else ``k{N}``.
    """
    if method == "resize":
        return "single"
    if method == "overview_only":
        return "k0"
    return f"k{num_patches}"


def default_vlm_metrics_path(manifest_path: str | Path, method: str, num_patches: int) -> Path:
    """Default per-method VLM CSV path including manifest stem.

    Example: ``outputs/metrics/vlm_metrics_docvqa_debug_bops_k2.csv``
    """
    stem = manifest_stem(manifest_path)
    suffix = vlm_patch_suffix(method, num_patches)
    return outputs_path("metrics", f"vlm_metrics_{stem}_{method}_{suffix}.csv")


def default_ocr_metrics_path(manifest_path: str | Path, experiment_stage: str) -> Path:
    """Default OCR metrics CSV path including manifest stem and stage."""
    stem = manifest_stem(manifest_path)
    return outputs_path("metrics", f"ocr_metrics_{stem}_{experiment_stage}.csv")


def default_ocr_checkpoint_path(manifest_path: str | Path, experiment_stage: str) -> Path:
    """Checkpoint JSON path for resumable OCR eval runs."""
    stem = manifest_stem(manifest_path)
    return outputs_path("checkpoints", f"ocr_eval_{stem}_{experiment_stage}.json")


def load_ocr_checkpoint(path: str | Path) -> dict[str, Any] | None:
    """Load OCR eval checkpoint if present."""
    p = Path(path)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_ocr_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    """Persist OCR eval checkpoint atomically."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(out)
    return out


def write_or_append_csv(
    rows: list[dict[str, Any]],
    path: str | Path,
    *,
    append: bool = False,
    overwrite: bool = False,
) -> Path:
    """Write result rows to CSV, optionally appending to an existing file.

    Args:
        rows: Result dicts to persist.
        path: Destination CSV path.
        append: If True, append rows (requires existing file or creates new).
        overwrite: If True and not appending, replace file contents.

    Returns:
        Resolved output path.

    Raises:
        FileExistsError: If file exists, not appending, and overwrite is False.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if not out.exists() and not append:
            out.write_text("", encoding="utf-8")
        return out

    fieldnames: list[str] = []
    seen: set[str] = set()
    existing_rows: list[dict[str, Any]] = []

    if out.exists() and append:
        with open(out, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                fieldnames = list(reader.fieldnames)
                seen = set(fieldnames)
            existing_rows = list(reader)

    for row in rows + existing_rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    if out.exists() and not append and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {out} without --overwrite. Use --append to add rows."
        )

    all_rows = existing_rows + rows if append else rows
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    return out


def filter_paper_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Filter metrics to rows eligible for paper tables.

    Keeps rows where dry_run, invalid_budget, and not_applicable are false,
    and experiment_stage is pilot or paper.
    """
    valid = df.copy()
    for col, default in (("dry_run", False), ("invalid_budget", False), ("not_applicable", False)):
        if col in valid.columns:
            valid = valid[valid[col].fillna(default).astype(bool) == False]
    if "experiment_stage" in valid.columns:
        valid = valid[valid["experiment_stage"].isin(PAPER_STAGES)]
    return valid


def serialize_answers(answers: list[str] | str) -> str:
    """JSON-serialize ground-truth answer list for CSV storage."""
    if isinstance(answers, str):
        answers = [answers]
    return json.dumps(answers, ensure_ascii=False)
