"""Tests for entity-row selection."""

from __future__ import annotations

from src.preprocessing.qe_bops_entity_row import extract_entity_field_tokens


def test_extract_khan_entity():
    parse = extract_entity_field_tokens("What is the no. of options held by S. H. Khan?")
    assert "khan" in parse.entity_tokens


def test_extract_field_tokens():
    parse = extract_entity_field_tokens("What was Final Wt of the child who had an initial Wt of 61.5 lbs?")
    assert "final" in parse.field_tokens or "wt" in parse.field_tokens
