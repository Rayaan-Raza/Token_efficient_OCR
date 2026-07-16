#!/usr/bin/env python3
"""Emit P14/P15/P16 gates for RAVEN-Select from existing or fresh n=500 eval."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd), flush=True)
    env = {**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONPATH": str(REPO)}
    subprocess.check_call(cmd, cwd=str(REPO), env=env)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--model", default="")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--skip-ablations", action="store_true")
    args = p.parse_args()

    py = sys.executable
    if not args.skip_eval:
        _run([
            py, "scripts/run_raven_select_eval.py",
            "--n", str(args.n),
            "--write-gates",
        ])
    if not args.skip_ablations:
        model = args.model
        if not model:
            overview = REPO / "outputs" / "metrics" / f"raven_select_overview_n{args.n}.json"
            if overview.exists():
                model = json.loads(overview.read_text(encoding="utf-8")).get("best_model") or "lgbm_reg"
            else:
                model = "lgbm_reg"
        _run([
            py, "scripts/run_raven_select_ablations.py",
            "--n", str(args.n),
            "--model", model,
        ])

    # Summarize gates
    gates = {}
    for name in ["P14_output_selector", "P15_significance", "P16_feature_ablations"]:
        path = REPO / "outputs" / "gates" / f"{name}.json"
        if path.exists():
            gates[name] = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps({k: {"passed": v.get("passed"), "message": v.get("message")} for k, v in gates.items()}, indent=2))


if __name__ == "__main__":
    main()
