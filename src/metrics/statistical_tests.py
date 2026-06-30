"""Statistical tests for paper claims."""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    diffs: list[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    arr = np.array(diffs, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    mean = float(arr.mean())
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boots.append(float(sample.mean()))
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return mean, lo, hi
