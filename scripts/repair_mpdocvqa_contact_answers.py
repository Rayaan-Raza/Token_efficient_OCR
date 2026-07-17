#!/usr/bin/env python3
"""Repair stringified MP-DocVQA contact-sheet answers and recompute VLM metrics."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.download_mpdocvqa_contact import _normalize_answers
from src.utils.experiment_io import serialize_answers, write_or_append_csv
from src.vlm.qa_metrics import anls, exact_match


def _parse_csv_answers(raw: object) -> list[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    text = str(raw)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return _normalize_answers(text)
    return _normalize_answers(parsed)


def repair_manifest(path: Path) -> int:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    changed = 0
    for row in rows:
        fixed = _normalize_answers(row.get("answer"))
        if fixed != row.get("answer"):
            changed += 1
        row["answer"] = fixed
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return changed


def repair_vlm_csv(path: Path, answers_by_id: dict[str, list[str]]) -> int:
    df = pd.read_csv(path)
    changed = 0
    for i, row in df.iterrows():
        image_id = str(row["image_id"])
        answers = answers_by_id.get(image_id) or _parse_csv_answers(row.get("ground_truth_answer"))
        serialized = serialize_answers(answers)
        pred = "" if pd.isna(row.get("parsed_prediction")) else str(row["parsed_prediction"])
        new_em = exact_match(pred, answers)
        new_anls = anls(pred, answers)
        if (
            str(row.get("ground_truth_answer")) != serialized
            or float(row.get("exact_match", -1)) != float(new_em)
            or abs(float(row.get("anls", -1)) - float(new_anls)) > 1e-12
        ):
            changed += 1
        df.at[i, "ground_truth_answer"] = serialized
        df.at[i, "exact_match"] = new_em
        df.at[i, "anls"] = new_anls
    write_or_append_csv(df.to_dict(orient="records"), path, overwrite=True)
    return changed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifests",
        default="Data/manifests/mpdocvqa_contact_300.jsonl,Data/manifests/mpdocvqa_contact_50.jsonl,Data/manifests/mpdocvqa_contact_2.jsonl",
    )
    p.add_argument(
        "--metrics-glob",
        default="outputs/metrics/vlm_metrics_mpdocvqa_contact_*.csv",
    )
    args = p.parse_args()

    answers_by_id: dict[str, list[str]] = {}
    for rel in [x.strip() for x in args.manifests.split(",") if x.strip()]:
        path = REPO_ROOT / rel
        if not path.exists():
            print(f"skip missing manifest: {path}")
            continue
        n = repair_manifest(path)
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for row in rows:
            answers_by_id[str(row["image_id"])] = list(row.get("answer") or [])
        print(f"manifest {path.name}: repaired {n}/{len(rows)} rows")
        print(f"  sample answer: {rows[0]['answer']!r}")

    for path in sorted((REPO_ROOT).glob(args.metrics_glob)):
        n = repair_vlm_csv(path, answers_by_id)
        print(f"metrics {path.name}: recomputed {n} rows")


if __name__ == "__main__":
    main()
