"""Tests for VLM metrics path helpers and CSV append behavior."""

import csv
from pathlib import Path

import pytest

from src.utils.experiment_io import (
    default_vlm_metrics_path,
    vlm_patch_suffix,
    write_or_append_csv,
)


def test_vlm_patch_suffix_resize():
    assert vlm_patch_suffix("resize", 4) == "single"


def test_vlm_patch_suffix_overview():
    assert vlm_patch_suffix("overview_only", 4) == "k0"


def test_vlm_default_path_includes_manifest_stem(tmp_path, monkeypatch):
    from src.utils import experiment_io

    monkeypatch.setattr(experiment_io, "outputs_path", lambda *parts: tmp_path.joinpath(*parts))
    p = default_vlm_metrics_path("data/manifests/docvqa_debug.jsonl", "bops", 2)
    assert p.name == "vlm_metrics_docvqa_debug_bops_k2.csv"


def test_write_or_append_csv_no_clobber(tmp_path):
    path = tmp_path / "out.csv"
    write_or_append_csv([{"a": 1}], path, overwrite=True)
    with pytest.raises(FileExistsError):
        write_or_append_csv([{"a": 2}], path, overwrite=False, append=False)


def test_write_or_append_csv_append(tmp_path):
    path = tmp_path / "out.csv"
    write_or_append_csv([{"a": 1, "b": "x"}], path, overwrite=True)
    write_or_append_csv([{"a": 2, "b": "y"}], path, append=True, overwrite=False)
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[1]["a"] == "2"
