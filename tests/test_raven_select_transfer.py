"""Tests for transfer gating and effective sample reporting."""

from src.answer_selection import evaluate


def _result(*, resize: bool, shortest: bool) -> dict:
    return {
        "n": 300,
        "requested_n": 300,
        "model": "raven_select_rule",
        "anls": 0.5,
        "em": 0.4,
        "beats_resize": resize,
        "beats_shortest_nonempty": shortest,
        "vs_resize": {
            "ci_lower_positive": resize,
            "ci95": [0.01, 0.03] if resize else [-0.01, 0.01],
        },
        "vs_shortest_nonempty": {
            "ci_lower_positive": shortest,
            "ci95": [0.01, 0.03] if shortest else [-0.01, 0.01],
        },
    }


def test_p18_transfer_statuses(monkeypatch):
    monkeypatch.setattr(evaluate, "write_gate_report", lambda *_: None)

    full = evaluate.write_p18_gate(
        _result(resize=True, shortest=True), dataset="infographicvqa"
    )
    partial = evaluate.write_p18_gate(
        _result(resize=True, shortest=False), dataset="infographicvqa"
    )
    shortest_only = evaluate.write_p18_gate(
        _result(resize=False, shortest=True), dataset="infographicvqa"
    )

    assert full.passed is True
    assert full.metrics["status"] == "FULL TRANSFER"
    assert partial.passed is False
    assert partial.metrics["status"] == "PARTIAL TRANSFER"
    assert shortest_only.metrics["status"] == "FAIL"
