#!/usr/bin/env python3
"""Resumable three-reader VLM driver with CUDA retry for journal scale runs."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, retries: int = 3) -> None:
    import os

    env = {**os.environ, "PYTHONPATH": str(REPO)}
    if "PYTORCH_CUDA_ALLOC_CONF" not in env:
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    for attempt in range(1, retries + 1):
        print(">>", " ".join(cmd), f"(attempt {attempt}/{retries})", flush=True)
        proc = subprocess.run(cmd, cwd=str(REPO), env=env)
        if proc.returncode == 0:
            return
        print(f"command failed with code {proc.returncode}", flush=True)
        if attempt < retries:
            time.sleep(30 * attempt)
    raise SystemExit(f"failed after {retries} attempts: {' '.join(cmd)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--manifest", default="Data/manifests/docvqa_1000.jsonl")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--metrics-tag", default="")
    p.add_argument("--methods", default="resize,bm25_only,learned_lgbm_strict")
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--retries", type=int, default=5)
    args = p.parse_args()

    py = sys.executable
    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        cmd = [
            py,
            "scripts/run_vlm_eval.py",
            "--manifest",
            args.manifest,
            "--method",
            method,
            "--num-patches",
            "2",
            "--limit",
            str(args.limit),
            "--experiment-stage",
            "pilot",
            "--checkpoint-every",
            str(args.checkpoint_every),
            "--model",
            args.model,
        ]
        if args.metrics_tag:
            cmd.extend(["--metrics-tag", args.metrics_tag])
        _run(cmd, retries=args.retries)

    print("All VLM methods complete.", flush=True)


if __name__ == "__main__":
    main()
