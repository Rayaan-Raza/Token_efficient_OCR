"""Validate JSONL manifest files."""

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    result = validate_manifest(Path(args.manifest))
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
