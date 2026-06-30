"""Repository path resolution utilities for the BOPS project.

This module centralizes all filesystem path logic so scripts and library code
never hard-code absolute paths. It supports the lowercase ``data/`` layout from
the research plan while transparently falling back to legacy ``Data/`` paths on
case-insensitive filesystems (e.g. Windows).

Constants:
    REPO_ROOT: Absolute path to the project root (parent of ``src/``).

Typical usage::

    from src.utils.paths import data_path, outputs_path, repo_path

    manifest = data_path("manifests", "textocr_debug.jsonl")
    out_csv = outputs_path("metrics", "ocr_metrics.csv")
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(*parts: str) -> Path:
    """Join path components under the repository root.

    Args:
        *parts: Relative path segments (e.g. ``"configs"``, ``"smoke_test.yaml"``).

    Returns:
        Absolute :class:`~pathlib.Path` under ``REPO_ROOT``.
    """
    return REPO_ROOT.joinpath(*parts)


def data_path(*parts: str) -> Path:
    """Resolve a path under ``data/``, with fallback to ``Data/``.

    On Linux the project uses lowercase ``data/``. On Windows the existing
    checkout may use ``Data/``; this helper returns whichever exists.

    Args:
        *parts: Segments after the data directory (e.g. ``"manifests"``, ``"foo.jsonl"``).

    Returns:
        Resolved data directory path (may not exist if neither variant is present).
    """
    lower = repo_path("data", *parts)
    if lower.exists() or not parts:
        return lower
    upper = repo_path("Data", *parts)
    return upper if upper.exists() else lower


def outputs_path(*parts: str) -> Path:
    """Resolve a path under ``outputs/`` and create parent directories.

    If the final component looks like a file (has a suffix), only parent
    directories are created so filenames are not mistaken for folders.

    Args:
        *parts: Segments under ``outputs/``.

    Returns:
        Absolute path; directories are created as needed.
    """
    p = repo_path("outputs", *parts)
    if p.suffix:
        p.parent.mkdir(parents=True, exist_ok=True)
    else:
        p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_dir(path: Path) -> Path:
    """Create a directory and all parents; no-op if it already exists.

    Args:
        path: Directory to create.

    Returns:
        The same ``path`` for chaining.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
