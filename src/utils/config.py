"""YAML configuration loader for experiment and smoke-test configs.

Configs live under ``configs/`` (e.g. ``smoke_test.yaml``, ``ocr_eval.yaml``).
Paths in config files may be relative to the repository root.

Example::

    from src.utils.config import load_config

    cfg = load_config("configs/smoke_test.yaml")
    area_ratio = cfg["area_ratio"]
"""

from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import repo_path


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file from an absolute or repo-relative path.

    Args:
        config_path: Path to ``.yaml`` / ``.yml`` file.

    Returns:
        Parsed mapping; empty dict if the file is empty.
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = repo_path(path)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
