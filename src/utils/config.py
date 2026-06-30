"""YAML configuration loader."""

from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import repo_path


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = repo_path(path)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
