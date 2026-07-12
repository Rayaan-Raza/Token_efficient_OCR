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
        return GateResult("G2_oracle", False, "min", {}, "oracle thresholds", "missing oracle_coverage_by_k.csv")

    by_k = {int(r["k"]): float(r["oracle_coverage"]) for r in rows if r.get("k")}
    o2 = by_k.get(2, 0.0)
    o4 = by_k.get(4, 0.0)
    metrics = {"oracle_at_2": o2, "oracle_at_4": o4}

    if o2 >= cfg["oracle_at_2_min"]:
        headline_k = 2
        passed = True
    elif o4 >= cfg["oracle_at_4_min"]:
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
            json.dump({"headline_k": headline_k, "oracle_at_2": o2, "oracle_at_4": o4}, f, indent=2)

    return GateResult(
        "G2_oracle",
        passed,
        "min",
        metrics,
        f"oracle@2>={cfg['oracle_at_2_min']} OR oracle@4>={cfg['oracle_at_4_min']}",
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
    path = coverage_csv or outputs_path("metrics", "coverage_by_method.csv")
    rows = _read_csv(path)
    if not rows:
        return GateResult("G3_heuristic", False, "min", {}, "coverage thresholds", "missing coverage_by_method.csv")

    hk = load_headline_k()
    by_method = {r["method"]: float(r["coverage"]) for r in rows if int(r.get("k", hk)) == hk}

    qe = by_method.get("qe_bops", 0.0)
    metrics = {"headline_k": hk, "qe_bops_coverage": qe}
    baselines = ["bops_fair_pool", "bops_qa_fair_pool", "bm25_only", "ocr_confidence_topk", "uniform"]
    beats_all = all(qe > by_method.get(b, 0.0) for b in baselines)
    metrics.update({f"coverage_{b}": by_method.get(b, 0.0) for b in baselines})

    floor = cfg_min["coverage_k2_min"] if hk == 2 else cfg_min["coverage_k4_min"]
    min_pass = beats_all and qe >= floor

    strong_target = cfg_str["coverage_k2_min"] if hk == 2 else cfg_str["coverage_k4_min"]
    strong_pass = qe >= strong_target

    return GateResult(
        "G3_heuristic",
        min_pass,
        "strong" if strong_pass else "min",
        metrics,
        f"beat all baselines @K={hk}, floor>={floor}; strong>={strong_target}",
        "strong pass" if strong_pass else ("min pass" if min_pass else "failed"),
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
}
