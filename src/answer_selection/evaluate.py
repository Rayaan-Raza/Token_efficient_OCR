"""Evaluation helpers and gate writers for RAVEN-Select."""

from __future__ import annotations

import json
from typing import Any

from src.extraction.gates import RevGateResult, write_gate_report
from src.utils.paths import outputs_path


def write_select_summary(
    result: dict[str, Any], *, tag: str = "", dataset: str = "docvqa"
) -> str:
    n = result["n"]
    model = result.get("model", "lgbm_reg")
    suffix = f"_{tag}" if tag else ""
    dataset_suffix = "" if dataset == "docvqa" else f"_{dataset}"
    path = outputs_path(
        "metrics", f"raven_select_{model}{dataset_suffix}_n{n}{suffix}.json"
    )
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


def write_p17_gate(result: dict[str, Any], *, n: int | None = None) -> RevGateResult:
    """P17 DocVQA scale gate: PASS / PARTIAL / FAIL vs resize and shortest.

    PASS: beats both with CI lower > 0.
    PARTIAL: significantly beats exactly one of resize / shortest_nonempty.
    FAIL: significantly beats neither.
    """
    n_eff = int(n if n is not None else result.get("n") or 0)
    vs_r = result.get("vs_resize", {})
    vs_s = result.get("vs_shortest_nonempty", {})
    beats_resize = bool(result.get("beats_resize")) and bool(vs_r.get("ci_lower_positive"))
    beats_short = bool(result.get("beats_shortest_nonempty")) and bool(
        vs_s.get("ci_lower_positive")
    )
    if beats_resize and beats_short:
        status = "PASS"
        passed = True
    elif beats_resize or beats_short:
        # One of the two significance targets holds (resize-only or shortest-only).
        status = "PARTIAL"
        passed = False
    else:
        status = "FAIL"
        passed = False
    gate = RevGateResult(
        name="P17_docvqa_scale",
        passed=passed,
        metrics={
            "n": n_eff,
            "status": status,
            "model": result.get("model"),
            "method_version": result.get("method_version") or (result.get("method") or {}).get("method_version"),
            "raven_select_anls": result.get("anls"),
            "raven_select_em": result.get("em"),
            "vs_resize": vs_r,
            "vs_shortest_nonempty": vs_s,
            "beats_resize_significant": beats_resize,
            "beats_shortest_significant": beats_short,
        },
        thresholds={
            "ci_lower_vs_resize_gt_0": True,
            "ci_lower_vs_shortest_gt_0": True,
            "n_min_for_full_validation": 1000,
        },
        message=(
            f"{status}: n={n_eff} anls={result.get('anls')} "
            f"vs_resize_ci={vs_r.get('ci95')} vs_short_ci={vs_s.get('ci95')}"
        ),
    )
    # Keep a size-specific report so n=500 / n=1000 / full do not overwrite each other.
    write_gate_report(f"P17_docvqa_scale_n{n_eff}", gate)
    # Also write the canonical name for the latest evaluated n.
    write_gate_report("P17_docvqa_scale", gate)
    return gate


def write_p18_gate(result: dict[str, Any], *, dataset: str) -> RevGateResult:
    """P18 frozen-rule transfer gate for a non-DocVQA dataset.

    FULL TRANSFER requires significant gains over resize and shortest.
    PARTIAL TRANSFER requires a significant gain over resize only.
    """
    if dataset == "docvqa":
        raise ValueError("P18 is defined only for transfer datasets")

    vs_r = result.get("vs_resize", {})
    vs_s = result.get("vs_shortest_nonempty", {})
    beats_resize = bool(result.get("beats_resize")) and bool(
        vs_r.get("ci_lower_positive")
    )
    beats_short = bool(result.get("beats_shortest_nonempty")) and bool(
        vs_s.get("ci_lower_positive")
    )
    if beats_resize and beats_short:
        status = "FULL TRANSFER"
        passed = True
    elif beats_resize:
        status = "PARTIAL TRANSFER"
        passed = False
    else:
        status = "FAIL"
        passed = False

    n_eff = int(result.get("n") or 0)
    gate = RevGateResult(
        name="P18_dataset_transfer",
        passed=passed,
        metrics={
            "dataset": dataset,
            "n": n_eff,
            "requested_n": result.get("requested_n"),
            "status": status,
            "model": result.get("model"),
            "method_version": result.get("method_version")
            or (result.get("method") or {}).get("method_version"),
            "raven_select_anls": result.get("anls"),
            "raven_select_em": result.get("em"),
            "vs_resize": vs_r,
            "vs_shortest_nonempty": vs_s,
            "beats_resize_significant": beats_resize,
            "beats_shortest_significant": beats_short,
        },
        thresholds={
            "ci_lower_vs_resize_gt_0": True,
            "ci_lower_vs_shortest_gt_0": True,
        },
        message=(
            f"{status}: dataset={dataset} n={n_eff} anls={result.get('anls')} "
            f"vs_resize_ci={vs_r.get('ci95')} vs_short_ci={vs_s.get('ci95')}"
        ),
    )
    write_gate_report(f"P18_dataset_transfer_{dataset}_n{n_eff}", gate)
    return gate


__all__ = [
    "write_p14_gate",
    "write_p15_gate",
    "write_p16_gate",
    "write_p17_gate",
    "write_p18_gate",
    "write_select_summary",
]
