#!/usr/bin/env python3
"""Heuristically label the RAVEN-Select taxonomy review sheet."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.answer_selection.dataset import _load_ocr_presence
from src.utils.paths import outputs_path


METHODS = ["resize", "bm25", "ler_bops"]
SUCCESS_LABELS = [f"S{i}" for i in range(1, 7)]
FAILURE_LABELS = [f"F{i}" for i in range(1, 9)]


@dataclass
class LabeledRecord:
    record: dict
    primary_label: str
    secondary_label: str | None


def _safe_json_loads(text: str) -> list[str]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [str(text)]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def _norm(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _pred_len(text: str) -> int:
    return len(_norm(text))


def _ocr_flags(ocr, image_id: str) -> dict[str, bool]:
    flags: dict[str, bool] = {}
    for m in METHODS:
        key = (image_id, m)
        if ocr is not None and key in ocr.index:
            row = ocr.loc[key]
            flags[m] = bool(
                row.get("pred_in_full_ocr", False) or row.get("pred_in_patch_ocr", False)
            )
        else:
            flags[m] = False
    return flags


def _prediction_map(record: dict) -> dict[str, str]:
    return {
        "resize": str(record.get("resize_pred", "")),
        "bm25": str(record.get("bm25_pred", "")),
        "ler_bops": str(record.get("ler_bops_pred", "")),
    }


def _label_record(record: dict, ocr) -> LabeledRecord:
    image_id = record["image_id"]
    preds = _prediction_map(record)
    gold_list = _safe_json_loads(str(record.get("gold", "")))
    gold_norms = {_norm(g) for g in gold_list if g}

    ocr_flags = _ocr_flags(ocr, image_id)
    raven_route = str(record.get("raven_select_route", "resize"))
    shortest_route = str(record.get("shortest_route", "resize"))

    raven_anls = float(record.get("raven_select_anls", 0.0))
    raven_em = float(record.get("raven_select_em", 0.0))
    resize_anls = float(record.get("resize_anls", 0.0))
    resize_em = float(record.get("resize_em", 0.0))
    shortest_anls = float(record.get("shortest_anls", 0.0))

    anls_by_route = {
        "resize": resize_anls,
        "bm25": float(record.get("bm25_anls", record.get("anls__bm25", 0.0)))
        if "bm25_anls" in record or "anls__bm25" in record
        else float(record.get("bm25_anls", 0.0)),
        "ler_bops": float(record.get("ler_bops_anls", record.get("anls__ler_bops", 0.0)))
        if "ler_bops_anls" in record or "anls__ler_bops" in record
        else float(record.get("ler_bops_anls", 0.0)),
    }

    # Failure heuristics (priority order).
    failure_rules: list[tuple[str, bool]] = [
        ("F1", all(v <= 1e-9 for v in anls_by_route.values()) and raven_anls <= 1e-9),
        ("F2", raven_anls >= 0.999 and not ocr_flags.get(raven_route, False)),
        (
            "F3",
            raven_anls <= 1e-9
            and ocr_flags.get(raven_route, False)
            and shortest_route == raven_route
            and any(v >= 0.999 for r, v in anls_by_route.items() if r != raven_route),
        ),
        (
            "F4",
            sum(v >= 0.999 for v in anls_by_route.values()) >= 2
            and len({_norm(preds[r]) for r, v in anls_by_route.items() if v >= 0.999}) >= 2,
        ),
        ("F5", raven_anls >= 0.9 and raven_em <= 1e-9 and ocr_flags.get(raven_route, False)),
        ("F6", 0.5 <= raven_anls < 0.999 and not ocr_flags.get(raven_route, False)),
        (
            "F7",
            0.2 <= raven_anls < 0.9
            and ocr_flags.get(raven_route, False)
            and any(_pred_len(preds[raven_route]) < _pred_len(g) for g in gold_norms),
        ),
        ("F8", len(gold_norms) > 1 and raven_anls < 0.999),
    ]

    matched_failures = [label for label, ok in failure_rules if ok]
    if matched_failures:
        primary = matched_failures[0]
        secondary = matched_failures[1] if len(matched_failures) > 1 else None
        return LabeledRecord(record, primary, secondary)

    # Success heuristics (priority order).
    success_rules: list[tuple[str, bool]] = [
        (
            "S3",
            raven_anls >= 0.999
            and raven_route in {"bm25", "ler_bops"}
            and resize_anls < 0.999,
        ),
        (
            "S4",
            raven_anls >= 0.999
            and raven_route == "resize"
            and shortest_route != "resize"
            and any(anls_by_route[r] < 0.999 for r in ("bm25", "ler_bops")),
        ),
        (
            "S2",
            raven_anls >= 0.999
            and resize_anls < 0.999
            and not ocr_flags.get("resize", False),
        ),
        (
            "S1",
            raven_anls >= 0.999
            and resize_anls < 0.999
            and _pred_len(preds[raven_route]) < _pred_len(preds["resize"]),
        ),
        ("S5", raven_anls >= 0.999 and resize_anls >= 0.9 and resize_em <= 1e-9 and raven_em >= 0.999),
        (
            "S6",
            raven_anls >= 0.999
            and raven_route != shortest_route
            and shortest_anls < 0.999,
        ),
    ]

    matched_success = [label for label, ok in success_rules if ok]
    if matched_success:
        primary = matched_success[0]
        secondary = matched_success[1] if len(matched_success) > 1 else None
        return LabeledRecord(record, primary, secondary)

    # Default fallback.
    if raven_anls >= 0.999:
        return LabeledRecord(record, "S6", None)
    return LabeledRecord(record, "F8", None)


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    input_path = outputs_path("labels", "raven_select_taxonomy_review_n500.jsonl")
    ocr = _load_ocr_presence(500)
    if ocr is None:
        raise FileNotFoundError("Missing OCR presence cache; run run_raven_select_eval.py --rebuild-ocr")

    labeled: list[dict] = []
    counts = Counter()
    secondary_counts = Counter()

    for record in _iter_jsonl(input_path):
        labeled_record = _label_record(record, ocr)
        record["primary_label"] = labeled_record.primary_label
        record["secondary_label"] = labeled_record.secondary_label or ""
        record["label_source"] = "heuristic_v1"
        labeled.append(record)

        counts[labeled_record.primary_label] += 1
        if labeled_record.secondary_label:
            secondary_counts[labeled_record.secondary_label] += 1

    output_jsonl = outputs_path("labels", "raven_select_taxonomy_labeled_n500.jsonl")
    _write_jsonl(output_jsonl, labeled)

    counts_path = outputs_path("metrics", "raven_select_taxonomy_counts_n500.json")
    counts_path.parent.mkdir(parents=True, exist_ok=True)
    with counts_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "primary": {k: counts.get(k, 0) for k in SUCCESS_LABELS + FAILURE_LABELS},
                "secondary": {k: secondary_counts.get(k, 0) for k in SUCCESS_LABELS + FAILURE_LABELS},
                "total": len(labeled),
            },
            f,
            indent=2,
        )

    # Qualitative examples.
    successes = [r for r in labeled if str(r.get("primary_label", "")).startswith("S")]
    failures = [r for r in labeled if str(r.get("primary_label", "")).startswith("F")]

    failure_picks: list[dict] = []
    for label in ["F1", "F2", "F3"]:
        match = next((r for r in failures if r.get("primary_label") == label and r not in failure_picks), None)
        if match is not None:
            failure_picks.append(match)
    for r in failures:
        if len(failure_picks) >= 3:
            break
        if r not in failure_picks:
            failure_picks.append(r)

    success_picks = successes[:3]

    def _example_payload(row: dict) -> dict:
        return {
            "image_id": row.get("image_id"),
            "primary_label": row.get("primary_label"),
            "secondary_label": row.get("secondary_label", ""),
            "question": row.get("question"),
            "gold": row.get("gold"),
            "raven_select_route": row.get("raven_select_route"),
            "raven_select_pred": row.get("raven_select_pred"),
        }

    examples = {
        "success_examples": [_example_payload(r) for r in success_picks],
        "failure_examples": [_example_payload(r) for r in failure_picks],
    }

    examples_path = outputs_path("metrics", "raven_select_qualitative_examples_n500.json")
    examples_path.parent.mkdir(parents=True, exist_ok=True)
    with examples_path.open("w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_jsonl),
                "counts": str(counts_path),
                "examples": str(examples_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
