"""Evaluation helpers and gate writers for RAVEN-Select."""

from __future__ import annotations

import json
from typing import Any

from src.extraction.gates import RevGateResult, write_gate_report
from src.utils.paths import outputs_path


def write_select_summary(result: dict[str, Any], *, tag: str = "") -> str:
    n = result["n"]
    model = result.get("model", "lgbm_reg")
    suffix = f"_{tag}" if tag else ""
    path = outputs_path("metrics", f"raven_select_{model}_n{n}{suffix}.json")
    # Drop large vectors from disk summary optionally keep them
    slim = {k: v for k, v in result.items() if k not in ("anls_vec", "em_vec")}
    path.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    return str(path)


def write_p14_gate(result: dict[str, Any]) -> RevGateResult:
    """P14: beat shortest_nonempty on n=500 (primary)."""
    passed = bool(result.get("beats_shortest_nonempty")) and result.get("n", 0) >= 500
    # Also allow n=300 reporting but gate is n=500
    if result.get("n") == 500:
        passed = bool(result.get("beats_shortest_nonempty"))
    else:
        passed = False  # P14 is defined on n=500
        msg_extra = "P14 requires n=500"
    short = result.get("baselines", {}).get("shortest_nonempty", {}).get("anls", None)
    gate = RevGateResult(
        name="P14_output_selector",
        passed=passed if result.get("n") == 500 else False,
        metrics={
            "n": result.get("n"),
            "model": result.get("model"),
            "raven_select_anls": result.get("anls"),
            "raven_select_em": result.get("em"),
            "shortest_nonempty_anls": short,
            "beats_shortest_nonempty": result.get("beats_shortest_nonempty"),
            "target_anls_met": result.get("target_anls_met"),
            "target_em_met": result.get("target_em_met"),
        },
        thresholds={"beats_shortest_nonempty": True, "n": 500},
        message=(
            "PASS" if (result.get("n") == 500 and result.get("beats_shortest_nonempty"))
            else f"select={result.get('anls'):.4f} vs shortest={short}"
        ),
    )
    write_gate_report("P14_output_selector", gate)
    return gate


def write_p15_gate(result: dict[str, Any]) -> RevGateResult:
    """P15: CI lower > 0 vs resize and vs shortest_nonempty."""
    vs_r = result.get("vs_resize", {})
    vs_s = result.get("vs_shortest_nonempty", {})
    passed = (
        result.get("n") == 500
        and bool(vs_r.get("ci_lower_positive"))
        and bool(vs_s.get("ci_lower_positive"))
    )
    gate = RevGateResult(
        name="P15_significance",
        passed=passed,
        metrics={
            "n": result.get("n"),
            "vs_resize": vs_r,
            "vs_shortest_nonempty": vs_s,
            "raven_select_anls": result.get("anls"),
        },
        thresholds={"ci_lower_vs_resize_gt_0": True, "ci_lower_vs_shortest_gt_0": True},
        message="PASS" if passed else f"vs_resize_ci={vs_r.get('ci95')} vs_short_ci={vs_s.get('ci95')}",
    )
    write_gate_report("P15_significance", gate)
    return gate


def write_p16_gate(full_anls: float, ablation_rows: list[dict[str, Any]]) -> RevGateResult:
    """P16: removing key feature groups should hurt (not improve) full selector."""
    harmful = {
        r["drop_group"]: r["anls"]
        for r in ablation_rows
        if r.get("drop_group") and r["anls"] > full_anls + 1e-6
    }
    hurt = {
        r["drop_group"]: r["anls"]
        for r in ablation_rows
        if r.get("drop_group") and r["anls"] < full_anls - 1e-6
    }
    # Pass if every ablation group hurts (or at least does not improve) — strict: none improve
    passed = len(harmful) == 0 and len(hurt) >= 3
    gate = RevGateResult(
        name="P16_feature_ablations",
        passed=passed,
        metrics={
            "full_anls": full_anls,
            "ablations": ablation_rows,
            "harmful_removals": harmful,
            "hurt_removals": hurt,
        },
        thresholds={"no_ablation_improves": True, "at_least_3_groups_hurt": True},
        message="PASS" if passed else f"harmful={harmful} hurt={list(hurt)}",
    )
    write_gate_report("P16_feature_ablations", gate)
    return gate


__all__ = [
    "write_p14_gate",
    "write_p15_gate",
    "write_p16_gate",
    "write_select_summary",
]
