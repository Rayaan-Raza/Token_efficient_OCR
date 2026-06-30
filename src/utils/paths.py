"""Repo-root-relative path helpers."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


def data_path(*parts: str) -> Path:
    """Resolve data/ with fallback to Data/ for legacy layout."""
    lower = repo_path("data", *parts)
    if lower.exists() or not parts:
        return lower
    upper = repo_path("Data", *parts)
    return upper if upper.exists() else lower


def outputs_path(*parts: str) -> Path:
    p = repo_path("outputs", *parts)
    if p.suffix:
        p.parent.mkdir(parents=True, exist_ok=True)
    else:
        p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
