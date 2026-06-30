"""JSONL manifest validation (Phase 2B gate).

Checks required fields, duplicate ``image_id`` values, and that every
``image_path`` resolves to an existing file on disk. Used before running
OCR/VLM experiments to catch broken manifests early.

CLI::

    python src/data/validate_manifest.py --manifest data/manifests/textocr_debug.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import data_path

REQUIRED_FIELDS = {"image_id", "dataset", "split", "image_path"}


def _resolve_image_path(ip: str) -> Path:
    """Resolve manifest ``image_path`` to an absolute filesystem path.

    Tries repo-relative path first, then ``data_path`` with ``data/`` prefix stripped.

    Args:
        ip: Path string from manifest row.

    Returns:
        Absolute :class:`~pathlib.Path` (may not exist).
    """
    p = Path(ip)
    if p.is_absolute():
        return p
    candidate = REPO_ROOT / ip
    if candidate.exists():
        return candidate
    parts = Path(ip).parts
    if parts and parts[0].lower() == "data":
        parts = parts[1:]
    return data_path(*parts)


def validate_manifest(manifest_path: Path) -> dict:
    """Validate a JSONL manifest file.

    Args:
        manifest_path: Path to ``.jsonl`` manifest.

    Returns:
        Dict with ``rows``, ``errors`` (list of messages), and ``valid`` bool.
    """
    rows = []
    errors = []
    seen_ids = set()

    with open(manifest_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: invalid JSON: {e}")
                continue
            rows.append(row)
            missing = REQUIRED_FIELDS - set(row.keys())
            if missing:
                errors.append(f"Line {i}: missing fields {missing}")
            iid = row.get("image_id")
            if iid in seen_ids:
                errors.append(f"Line {i}: duplicate image_id {iid}")
            seen_ids.add(iid)
            ip = row.get("image_path", "")
            p = _resolve_image_path(ip)
            if not p.exists():
                errors.append(f"Line {i}: image not found {ip}")

    return {
        "manifest": str(manifest_path),
        "rows": len(rows),
        "errors": errors,
        "valid": len(errors) == 0,
    }


def main() -> None:
    """CLI entry: print JSON result and exit 1 if invalid."""
    parser = argparse.ArgumentParser(description="Validate a BOPS JSONL manifest.")
    parser.add_argument("--manifest", required=True, help="Path to .jsonl manifest")
    args = parser.parse_args()
    result = validate_manifest(Path(args.manifest))
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
