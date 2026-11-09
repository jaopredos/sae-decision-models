"""Statistical helpers shared by run_phaseA.py / run_phaseB.py.

Operates on plain Python/NumPy, taking the per-decision records logged by src/runner.py
(one row per (example, condition)) and computing the aggregates the REPORT.md needs.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence

import numpy as np


def entropy(probs: Dict[str, float]) -> float:
    vals = np.array(list(probs.values()), dtype=float)
    vals = vals[vals > 0]
    return float(-(vals * np.log(vals)).sum())


def flip_rate(records_by_example: Dict[str, List[dict]]) -> float:
    """records_by_example: example_id -> list of records (one per order/condition tested),
    each with 'chosen_id'. Flip = at least two different chosen_id across the records."""
    if not records_by_example:
        return float("nan")
    flips = [len({r["chosen_id"] for r in recs}) > 1 for recs in records_by_example.values()]
    return float(np.mean(flips))


def accuracy(records: Sequence[dict]) -> float:
    if not records:
        return float("nan")
    return float(np.mean([r["chosen_id"] == r["gold_id"] for r in records]))


def accuracy_by_position(records: Sequence[dict]) -> Dict[int, float]:
    by_pos = defaultdict(list)
    for r in records:
        by_pos[r["gold_position"]].append(r["chosen_id"] == r["gold_id"])
    return {pos: float(np.mean(v)) for pos, v in sorted(by_pos.items())}


def position_choice_distribution(records: Sequence[dict]) -> Dict[int, float]:
    """P(choose position i), i.e. how often the chosen option sat at each presented slot,
    vs. the uniform distribution expected under no positional bias."""
    counts = Counter(r["chosen_position"] for r in records)
    total = sum(counts.values())
    return {pos: counts[pos] / total for pos in sorted(counts)}


def bootstrap_ci(records: Sequence[dict], statistic_fn, n_boot: int = 1000, seed: int = 0,
                  alpha: float = 0.05):
    """Resample examples (not individual rows) with replacement, n_boot times, and return
    (point_estimate, lo, hi) of statistic_fn(records) under the empirical bootstrap."""
    if not records:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    n = len(records)
    point = statistic_fn(records)
    values = []
    idx_pool = np.arange(n)
    for _ in range(n_boot):
        idx = rng.choice(idx_pool, size=n, replace=True)
        sample = [records[i] for i in idx]
        values.append(statistic_fn(sample))
    lo = float(np.percentile(values, 100 * alpha / 2))
    hi = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return float(point), lo, hi


def bootstrap_ci_by_example(records_by_example: Dict[str, List[dict]], statistic_fn,
                             n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Like bootstrap_ci, but resamples whole examples (groups of records), appropriate for
    statistics like flip_rate that are naturally defined per example."""
    keys = list(records_by_example.keys())
    if not keys:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    point = statistic_fn(records_by_example)
    values = []
    for _ in range(n_boot):
        sample_keys = rng.choice(keys, size=len(keys), replace=True)
        sample = {f"{k}__{i}": records_by_example[k] for i, k in enumerate(sample_keys)}
        values.append(statistic_fn(sample))
    lo = float(np.percentile(values, 100 * alpha / 2))
    hi = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return float(point), lo, hi


def ece(records: Sequence[dict], n_bins: int = 15) -> float:
    """Expected calibration error using max-probability confidence vs correctness."""
    if not records:
        return float("nan")
    confs = np.array([max(r["probabilities"].values()) for r in records])
    correct = np.array([r["chosen_id"] == r["gold_id"] for r in records], dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(records)
    err = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confs > lo) & (confs <= hi) if lo > 0 else (confs >= lo) & (confs <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = confs[mask].mean()
        bin_acc = correct[mask].mean()
        err += (mask.sum() / total) * abs(bin_conf - bin_acc)
    return float(err)


def high_confidence_error_rate(records: Sequence[dict], threshold: float = 0.9) -> float:
    if not records:
        return float("nan")
    hc = [r for r in records if max(r["probabilities"].values()) > threshold]
    if not hc:
        return 0.0
    return float(np.mean([r["chosen_id"] != r["gold_id"] for r in hc]))


def majority_class_baseline(records: Sequence[dict]) -> float:
    golds = [r["gold_id"] for r in records]
    if not golds:
        return float("nan")
    top = Counter(golds).most_common(1)[0][1]
    return top / len(golds)


def random_baseline(records: Sequence[dict]) -> float:
    """Expected accuracy of a uniform-random choice over the options actually presented."""
    vals = [1.0 / len(r["probabilities"]) for r in records if r["probabilities"]]
    return float(np.mean(vals)) if vals else float("nan")


def contextual_calibration(probs: Dict[str, float], prior: Dict[str, float]) -> Dict[str, float]:
    """Divide by content-free prior probabilities (per id) and renormalize."""
    adjusted = {k: probs.get(k, 0.0) / max(prior.get(k, 1e-8), 1e-8) for k in probs}
    total = sum(adjusted.values()) or 1.0
    return {k: v / total for k, v in adjusted.items()}


def rotation_average_probs(records_same_example: Sequence[dict]) -> Dict[str, float]:
    """Average probabilities across rotation/order conditions, per id (AnyJev-style), then
    the caller can take argmax for the aggregated decision."""
    totals: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    for r in records_same_example:
        for k, v in r["probabilities"].items():
            totals[k] += v
            counts[k] += 1
    return {k: totals[k] / counts[k] for k in totals}


def argmax_id(probs: Dict[str, float]) -> str:
    return max(probs, key=probs.get)
