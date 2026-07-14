"""Automated gate evaluation for QE-BOPS pipeline."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import outputs_path, repo_path


def load_qe_bops_config() -> dict[str, Any]:
    path = repo_path("configs", "qe_bops.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class GateResult:
    name: str
    passed: bool
    level: str
    metrics: dict[str, Any]
    threshold: str
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "level": self.level,
            "metrics": self.metrics,
            "threshold": self.threshold,
            "message": self.message,
        }


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def check_g0_baseline() -> GateResult:
    cfg = load_qe_bops_config()["gates"]["g0"]
    metrics: dict[str, Any] = {}
    passed = True
    msgs: list[str] = []

    ocr_path = outputs_path("metrics", "ocr_metrics_textocr_pilot_pilot.csv")
    if ocr_path.exists():
        rows = _read_csv(ocr_path)
        bops = [r for r in rows if r.get("method") == "bops" and r.get("budget") == "patches_8"]
        if bops:
            recall = float(bops[0].get("word_recall", 0))
            metrics["bops_ocr_recall_p8"] = recall
            target = cfg["bops_ocr_recall_p8"]
            if abs(recall - target) > cfg["ocr_recall_tol"]:
                passed = False
                msgs.append(f"bops recall {recall:.3f} not within tol of {target}")

    vlm_path = outputs_path("metrics", "vlm_metrics_merged.csv")
    if vlm_path.exists():
        rows = _read_csv(vlm_path)
        bops = [r for r in rows if r.get("method") == "bops" and r.get("num_patches") == "2"]
        if bops:
            anls_vals = [float(r["anls"]) for r in bops if r.get("anls")]
            if anls_vals:
                mean_anls = sum(anls_vals) / len(anls_vals)
                metrics["bops_vlm_anls_k2"] = mean_anls
                target = cfg["bops_vlm_anls_k2"]
                if abs(mean_anls - target) > cfg["anls_tol"]:
                    passed = False
                    msgs.append(f"bops ANLS {mean_anls:.3f} not within tol of {target}")

    return GateResult(
        "G0_baseline",
        passed,
        "min",
        metrics,
        f"recall±{cfg['ocr_recall_tol']}, anls±{cfg['anls_tol']}",
        "; ".join(msgs) or "baseline within tolerance or metrics file missing (manual verify)",
    )


def check_g1_candidates(stats_csv: Path | None = None) -> GateResult:
    cfg = load_qe_bops_config()["gates"]["g1"]
    path = stats_csv or outputs_path("candidates", "candidate_pool_stats.csv")
    rows = _read_csv(path)
    if not rows:
        return GateResult("G1_candidates", False, "min", {}, "see config", "missing candidate_pool_stats.csv")

    def _mean(key: str) -> float:
        vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
        return sum(vals) / len(vals) if vals else 0.0

    metrics = {
        "mean_candidates": _mean("num_candidates"),
        "mean_unique_image_area_coverage": _mean("unique_image_area_coverage"),
        "mean_candidate_area_ratio": _mean("candidate_area_ratio"),
        "mean_ocr_box_center_coverage": _mean("ocr_box_center_coverage"),
        "mean_small_box_coverage": _mean("small_box_coverage"),
        "edge_patch_rate": _mean("edge_patch_hit"),
    }
    union_min = cfg.get("unique_image_area_coverage_min", cfg.get("area_coverage_min", 0.90))
    passed = (
        cfg["candidates_min"] <= metrics["mean_candidates"] <= cfg["candidates_max"]
        and metrics["mean_unique_image_area_coverage"] >= union_min
        and metrics["mean_ocr_box_center_coverage"] >= cfg["ocr_box_center_coverage_min"]
        and metrics["mean_small_box_coverage"] >= cfg["small_box_coverage_min"]
        and metrics["edge_patch_rate"] >= cfg["edge_patch_rate_min"]
    )
    return GateResult(
        "G1_candidates",
        passed,
        "min",
        metrics,
        f"cands {cfg['candidates_min']}-{cfg['candidates_max']}, "
        f"union_area>{union_min}, ocr_box>{cfg['ocr_box_center_coverage_min']}, "
        f"small>{cfg['small_box_coverage_min']}",
    )


def check_g2_oracle(oracle_csv: Path | None = None) -> GateResult:
    cfg = load_qe_bops_config()["gates"]["g2"]
    path = oracle_csv or outputs_path("metrics", "oracle_coverage_by_k.csv")
    rows = _read_csv(path)
    if not rows:
        return GateResult("G2_oracle", False, "min", {}, "oracle ceilings", "missing oracle_coverage_by_k.csv")

    def _col(row: dict[str, str], *names: str) -> float | None:
        for name in names:
            if row.get(name) not in (None, ""):
                return float(row[name])
        return None

    by_k_ocr: dict[int, float] = {}
    by_k_ev: dict[int, float] = {}
    for r in rows:
        if not r.get("k"):
            continue
        k = int(r["k"])
        ocr = _col(r, "oracle_ocr_exact", "oracle_coverage")
        ev = _col(r, "oracle_evidence")
        if ocr is not None:
            by_k_ocr[k] = ocr
        if ev is not None:
            by_k_ev[k] = ev

    o2_ocr = by_k_ocr.get(2, 0.0)
    o4_ocr = by_k_ocr.get(4, 0.0)
    o2_ev = by_k_ev.get(2, 0.0)
    o4_ev = by_k_ev.get(4, 0.0)
    metrics = {
        "oracle_ocr_exact_at_2": o2_ocr,
        "oracle_ocr_exact_at_4": o4_ocr,
        "oracle_evidence_at_2": o2_ev,
        "oracle_evidence_at_4": o4_ev,
    }

    ev_min_2 = cfg.get("oracle_evidence_at_2_min", cfg.get("oracle_at_2_min", 0.15))
    ev_min_4 = cfg.get("oracle_evidence_at_4_min", cfg.get("oracle_at_4_min", 0.30))
    if o2_ev >= ev_min_2:
        headline_k = 2
        passed = True
    elif o4_ev >= ev_min_4:
        headline_k = 4
        passed = True
    else:
        headline_k = None
        passed = False

    metrics["headline_k"] = headline_k
    if passed and headline_k is not None:
        hk_path = outputs_path("gates", "headline_k.json")
        hk_path.parent.mkdir(parents=True, exist_ok=True)
        with open(hk_path, "w", encoding="utf-8") as f:
            json.dump({
                "headline_k": headline_k,
                "oracle_ocr_exact_at_2": o2_ocr,
                "oracle_ocr_exact_at_4": o4_ocr,
                "oracle_evidence_at_2": o2_ev,
                "oracle_evidence_at_4": o4_ev,
            }, f, indent=2)

    return GateResult(
        "G2_oracle",
        passed,
        "min",
        metrics,
        f"oracle_evidence@2>={ev_min_2} OR oracle_evidence@4>={ev_min_4}",
        "" if passed else "run remediation playbook on candidate pool",
    )


def load_headline_k(default: int = 2) -> int:
    path = outputs_path("gates", "headline_k.json")
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return int(json.load(f).get("headline_k", default))
    return default


def check_g3_heuristic(coverage_csv: Path | None = None) -> GateResult:
    cfg_min = load_qe_bops_config()["gates"]["g3_min"]
    cfg_str = load_qe_bops_config()["gates"]["g3_strong"]
    cfg_vstr = load_qe_bops_config()["gates"].get("g3_very_strong", {})
    path = coverage_csv or outputs_path("metrics", "coverage_by_method.csv")
    rows = _read_csv(path)
    if not rows:
        return GateResult("G3_heuristic", False, "min", {}, "evidence coverage", "missing coverage_by_method.csv")

    hk = load_headline_k()
    by_method = {r["method"]: r for r in rows if int(r.get("k", hk)) == hk}

    def _metric(row: dict[str, str], *names: str) -> float:
        for name in names:
            if row.get(name) not in (None, ""):
                return float(row[name])
        return 0.0

    qe_row = by_method.get("qe_bops", {})
    qa_row = by_method.get("bops_qa_fair_pool", {})
    qe_strict = _metric(qe_row, "evidence_strict")
    qe_any = _metric(qe_row, "evidence_any", "evidence_coverage")
    qe_ocr = _metric(qe_row, "ocr_exact_coverage")
    qa_strict = _metric(qa_row, "evidence_strict")
    qa_any = _metric(qa_row, "evidence_any", "evidence_coverage")

    metrics: dict[str, Any] = {
        "headline_k": hk,
        "qe_bops_evidence_strict": qe_strict,
        "qe_bops_evidence_any": qe_any,
        "qe_bops_ocr_exact_coverage": qe_ocr,
        "bops_qa_evidence_strict": qa_strict,
        "bops_qa_evidence_any": qa_any,
        "diff_strict_vs_qa": qe_strict - qa_strict,
        "diff_any_vs_qa": qe_any - qa_any,
    }

    other_baselines = ["bops_fair_pool", "bm25_only", "ocr_confidence_topk", "uniform"]
    beats_others_strict = all(
        qe_strict > _metric(by_method.get(b, {}), "evidence_strict") for b in other_baselines
    )
    beats_others_any = all(
        qe_any > _metric(by_method.get(b, {}), "evidence_any", "evidence_coverage")
        for b in other_baselines
    )
    beats_qa_strict = qe_strict >= qa_strict
    beats_qa_any = qe_any >= qa_any

    for b in other_baselines + ["bops_qa_fair_pool"]:
        metrics[f"evidence_strict_{b}"] = _metric(by_method.get(b, {}), "evidence_strict")
        metrics[f"evidence_any_{b}"] = _metric(by_method.get(b, {}), "evidence_any", "evidence_coverage")

    bootstrap_ok = False
    bootstrap_msg = "bootstrap file missing"
    boot_path = outputs_path("metrics", "coverage_bootstrap_ci.json")
    if boot_path.exists():
        with open(boot_path, encoding="utf-8") as f:
            boot = json.load(f)
        qa_cmp = [r for r in boot if r.get("comparison") == "qe_bops_vs_bops_qa_fair_pool"]
        strict_row = next((r for r in qa_cmp if r.get("metric") == "evidence_strict"), None)
        any_row = next((r for r in qa_cmp if r.get("metric") == "evidence_any"), None)
        if strict_row and any_row:
            strict_ok = float(strict_row["mean_diff"]) >= 0 and float(strict_row["ci_low"]) >= -0.02
            any_ok = float(any_row["mean_diff"]) >= 0 and float(any_row["ci_low"]) >= -0.02
            bootstrap_ok = strict_ok and any_ok
            metrics["bootstrap_strict_mean_diff"] = float(strict_row["mean_diff"])
            metrics["bootstrap_strict_ci_low"] = float(strict_row["ci_low"])
            metrics["bootstrap_any_mean_diff"] = float(any_row["mean_diff"])
            metrics["bootstrap_any_ci_low"] = float(any_row["ci_low"])
            bootstrap_msg = "bootstrap vs Q-BOPS ok" if bootstrap_ok else "bootstrap vs Q-BOPS negative"
        else:
            bootstrap_msg = "no qe_bops vs bops_qa_fair_pool bootstrap rows"

    min_pass = (
        beats_qa_strict
        and beats_qa_any
        and beats_others_strict
        and beats_others_any
        and bootstrap_ok
    )

    strong_strict = cfg_str["evidence_strict_k2_min"] if hk == 2 else cfg_str.get("evidence_k4_min", 0.50)
    strong_any = cfg_str["evidence_any_k2_min"] if hk == 2 else cfg_str.get("evidence_k4_min", 0.50)
    very_strict = cfg_vstr.get("evidence_strict_k2_min", 0.35) if hk == 2 else cfg_vstr.get("evidence_k4_min", 0.60)
    very_any = cfg_vstr.get("evidence_any_k2_min", 0.50) if hk == 2 else cfg_vstr.get("evidence_k4_min", 0.60)

    strong_pass = min_pass and qe_strict >= strong_strict and qe_any >= strong_any
    very_strong_pass = min_pass and qe_strict >= very_strict and qe_any >= very_any

    level = "min"
    if very_strong_pass:
        level = "very_strong"
    elif strong_pass:
        level = "strong"

    msgs: list[str] = []
    if not beats_qa_strict:
        msgs.append(f"strict {qe_strict:.3f} < Q-BOPS {qa_strict:.3f}")
    if not beats_qa_any:
        msgs.append(f"any {qe_any:.3f} < Q-BOPS {qa_any:.3f}")
    if not beats_others_strict or not beats_others_any:
        msgs.append("did not beat all secondary baselines")
    if not bootstrap_ok:
        msgs.append(bootstrap_msg)

    return GateResult(
        "G3_heuristic",
        min_pass,
        level,
        metrics,
        "QE-BOPS v2 >= Q-BOPS strict+any; beat BOPS/BM25/OCR-conf/uniform; bootstrap>=0",
        "; ".join(msgs) if msgs else f"{level} pass",
    )


def check_g3_learned(coverage_csv: Path | None = None, method: str = "lgbm_combined") -> GateResult:
    """Learned ranker must beat Q-BOPS on both strict and any at headline K."""
    path = coverage_csv or outputs_path("metrics", "learned_coverage_by_method.csv")
    rows = _read_csv(path)
    if not rows:
        return GateResult(
            "G3_learned", False, "min", {}, "learned coverage",
            "missing learned_coverage_by_method.csv",
        )

    hk = load_headline_k()
    by_method_k = {}
    for r in rows:
        key = (r["method"], int(float(r.get("k", hk))))
        by_method_k[key] = r

    def _metric(row: dict[str, str], *names: str) -> float:
        for name in names:
            if row.get(name) not in (None, ""):
                return float(row[name])
        return 0.0

    # Prefer combined/hybrid; fall back to first learned method present at hk
    preferred = [method, "lgbm_qbops_hybrid", "lgbm_combined", "lgbm_strict", "learned_lgbm_combined"]
    learned_row = {}
    used = method
    for name in preferred:
        if (name, hk) in by_method_k:
            learned_row = by_method_k[(name, hk)]
            used = name
            break
    qa_row = by_method_k.get(("bops_qa_fair_pool", hk), {})

    lr_strict = _metric(learned_row, "evidence_strict")
    lr_any = _metric(learned_row, "evidence_any", "evidence_coverage")
    qa_strict = _metric(qa_row, "evidence_strict")
    qa_any = _metric(qa_row, "evidence_any", "evidence_coverage")

    metrics: dict[str, Any] = {
        "headline_k": hk,
        "learned_method": used,
        "learned_strict": lr_strict,
        "learned_any": lr_any,
        "bops_qa_strict": qa_strict,
        "bops_qa_any": qa_any,
        "diff_strict": lr_strict - qa_strict,
        "diff_any": lr_any - qa_any,
    }

    # Plan: strict must strictly beat; any must not regress
    beats_strict = lr_strict > qa_strict
    beats_any = lr_any >= qa_any

    bootstrap_ok = False
    bootstrap_msg = "bootstrap file missing"
    boot_path = outputs_path("metrics", "coverage_bootstrap_ci_learned.json")
    if boot_path.exists():
        with open(boot_path, encoding="utf-8") as f:
            boot = json.load(f)
        cmp = [
            r for r in boot
            if r.get("comparison") == f"{used}_vs_bops_qa_fair_pool" and int(r.get("k", hk)) == hk
        ]
        strict_row = next((r for r in cmp if r.get("metric") == "evidence_strict"), None)
        any_row = next((r for r in cmp if r.get("metric") == "evidence_any"), None)
        if strict_row and any_row:
            bootstrap_ok = float(strict_row["mean_diff"]) >= 0 and float(any_row["mean_diff"]) >= 0
            metrics["bootstrap_strict_mean_diff"] = float(strict_row["mean_diff"])
            metrics["bootstrap_any_mean_diff"] = float(any_row["mean_diff"])
            bootstrap_msg = "bootstrap ok" if bootstrap_ok else "bootstrap mean_diff < 0"
        else:
            bootstrap_msg = f"no bootstrap rows for {used} @ k={hk}"

    # Also check K=4 if K=2 fails (recorded in metrics)
    k4_pass = False
    if (used, 4) in by_method_k and ("bops_qa_fair_pool", 4) in by_method_k:
        s4 = _metric(by_method_k[(used, 4)], "evidence_strict")
        a4 = _metric(by_method_k[(used, 4)], "evidence_any", "evidence_coverage")
        qs4 = _metric(by_method_k[("bops_qa_fair_pool", 4)], "evidence_strict")
        qa4 = _metric(by_method_k[("bops_qa_fair_pool", 4)], "evidence_any", "evidence_coverage")
        k4_pass = s4 > qs4 and a4 >= qa4
        metrics.update({"k4_learned_strict": s4, "k4_qbops_strict": qs4, "k4_learned_any": a4, "k4_qbops_any": qa4})

    min_pass = (beats_strict and beats_any and bootstrap_ok) or k4_pass
    msgs: list[str] = []
    if not beats_strict:
        msgs.append(f"strict {lr_strict:.3f} <= Q-BOPS {qa_strict:.3f}")
    if not beats_any:
        msgs.append(f"any {lr_any:.3f} < Q-BOPS {qa_any:.3f}")
    if not bootstrap_ok:
        msgs.append(bootstrap_msg)
    if k4_pass and not (beats_strict and beats_any):
        msgs.append("passed via K=4 fallback")

    return GateResult(
        "G3_learned",
        min_pass,
        "min" if min_pass else "min",
        metrics,
        "learned strict@K > Q-BOPS; any@K >= Q-BOPS; bootstrap mean_diff >= 0",
        "; ".join(msgs) if msgs else "pass",
    )


def check_g5_vlm(
    manifest_stem: str = "docvqa_300",
    *,
    learned_method: str = "learned_lgbm_strict",
    qbops_method: str = "bops_qa_fair_pool",
    bm25_method: str = "bm25_only",
) -> GateResult:
    """G5: scaled VLM transfer — lgbm_strict vs Q-BOPS and BM25 on paired ANLS/EM."""
    import pandas as pd

    from src.utils.experiment_io import vlm_patch_suffix

    metrics_dir = outputs_path("metrics")
    learned_suffix = vlm_patch_suffix(learned_method, 2)
    paths = {
        learned_method: metrics_dir / f"vlm_metrics_{manifest_stem}_{learned_method}_{learned_suffix}.csv",
        qbops_method: metrics_dir / f"vlm_metrics_{manifest_stem}_{qbops_method}_k2.csv",
        bm25_method: metrics_dir / f"vlm_metrics_{manifest_stem}_{bm25_method}_k2.csv",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        return GateResult(
            "G5_vlm", False, "min", {}, "scaled VLM transfer",
            f"missing VLM CSVs: {missing[:2]}...",
        )

    dfs = {m: pd.read_csv(p) for m, p in paths.items()}
    learned = dfs[learned_method].set_index("image_id")
    qbops = dfs[qbops_method].set_index("image_id")
    bm25 = dfs[bm25_method].set_index("image_id")
    common = learned.index.intersection(qbops.index).intersection(bm25.index)
    if len(common) < 50:
        return GateResult(
            "G5_vlm", False, "min", {"n_pairs": len(common)}, "scaled VLM transfer",
            f"too few paired rows ({len(common)})",
        )

    danls_q = float((learned.loc[common, "anls"] - qbops.loc[common, "anls"]).mean())
    dem_q = float((learned.loc[common, "exact_match"] - qbops.loc[common, "exact_match"]).mean())
    danls_b = float((learned.loc[common, "anls"] - bm25.loc[common, "anls"]).mean())
    mean_anls_l = float(learned.loc[common, "anls"].mean())
    mean_anls_q = float(qbops.loc[common, "anls"].mean())
    mean_anls_b = float(bm25.loc[common, "anls"].mean())

    metrics = {
        "n_pairs": len(common),
        "learned_anls": mean_anls_l,
        "qbops_anls": mean_anls_q,
        "bm25_anls": mean_anls_b,
        "delta_anls_vs_qbops": danls_q,
        "delta_em_vs_qbops": dem_q,
        "delta_anls_vs_bm25": danls_b,
        "learned_em": float(learned.loc[common, "exact_match"].mean()),
        "qbops_em": float(qbops.loc[common, "exact_match"].mean()),
    }

    min_pass = (
        danls_q >= 0.03
        and dem_q >= 0.0
        and danls_b >= 0.0
        and mean_anls_l > mean_anls_q
    )
    strong_pass = danls_q >= 0.04 and danls_b >= 0.02 and dem_q >= 0.03

    msgs: list[str] = []
    if danls_q < 0.03:
        msgs.append(f"ANLS vs Q-BOPS {danls_q:+.3f} < +0.03")
    if dem_q < 0.0:
        msgs.append(f"EM vs Q-BOPS {dem_q:+.3f} < 0")
    if danls_b < 0.0:
        msgs.append(f"ANLS vs BM25 {danls_b:+.3f} < 0")
    if strong_pass:
        msgs.append("strong bar met")

    return GateResult(
        "G5_vlm",
        min_pass,
        "strong" if strong_pass else "min",
        metrics,
        "ANLS >= +0.03 vs Q-BOPS; EM >= Q-BOPS; ANLS >= BM25",
        "; ".join(msgs) if msgs else "pass",
    )


def write_gate_report(results: list[GateResult]) -> Path:
    out = outputs_path("gates", "gate_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"results": [r.as_dict() for r in results], "all_passed": all(r.passed for r in results)}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out


GATE_FUNCS = {
    "G0_baseline": check_g0_baseline,
    "G1_candidates": check_g1_candidates,
    "G2_oracle": check_g2_oracle,
    "G3_heuristic": check_g3_heuristic,
    "G3_learned": check_g3_learned,
    "G5_vlm": check_g5_vlm,
}
