"""Seeded scenario bootstrap; paired differences preserve common random numbers."""
import numpy as np


def summarize(values, seed: int = 0, resamples: int = 2000) -> dict:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("Statistics require a finite, nonempty vector")
    rng = np.random.default_rng(seed)
    estimates = np.asarray([rng.choice(x, size=len(x), replace=True).mean() for _ in range(resamples)])
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"n": len(x), "mean": float(x.mean()), "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "ci95_low": float(low), "ci95_high": float(high)}


def paired_difference(first: dict[int, float], second: dict[int, float], seed: int = 0) -> dict:
    if first.keys() != second.keys():
        raise ValueError("Paired comparisons require identical scenario IDs")
    return summarize([first[k] - second[k] for k in sorted(first)], seed)
