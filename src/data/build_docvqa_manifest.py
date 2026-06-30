"""Build DocVQA subset manifests from the Hugging Face export (Phase 2B).

Slices an existing ``docvqa_val_500.jsonl`` (or similar) into smaller debug/pilot
manifests for fast iteration.

CLI::

    python src/data/build_docvqa_manifest.py --out data/manifests/docvqa_debug.jsonl --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import data_path, ensure_dir


def build_subset(source: Path, out_path: Path, limit: int) -> int:
    """Copy the first ``limit`` rows from a DocVQA manifest.

    Args:
        source: Source JSONL (full validation export).
        out_path: Destination JSONL.
        limit: Maximum number of QA samples.

    Returns:
        Number of rows written.
    """
    rows = []
    with open(source, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    ensure_dir(out_path.parent)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows[:limit]:
            row["split"] = "val"
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return min(limit, len(rows))


def main() -> None:
    """CLI: slice DocVQA manifest."""
    parser = argparse.ArgumentParser(description="Build DocVQA subset manifest.")
    parser.add_argument("--source", default=str(data_path("manifests", "docvqa_val_500.jsonl")))
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    n = build_subset(Path(args.source), Path(args.out), args.limit)
    print(f"Wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
