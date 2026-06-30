"""Unified manifest iterator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def load_manifest(manifest_path: str | Path) -> list[dict]:
    rows = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def iter_manifest(manifest_path: str | Path) -> Iterator[dict]:
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
