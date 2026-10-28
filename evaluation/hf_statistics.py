"""Paired, case-clustered bootstrap: compare the same questions, not raw means.

A confidence interval does not validate reference labels or model factuality.
This module never supplies fabricated observations.
"""
from __future__ import annotations

import math
import random
from statistics import mean
from typing import Any


ALLOWED_METRICS = {
    "exact_match", "answer_token_f1", "rouge_l_f1",
    "evidence_recall", "evidence_precision",
    "evidence_mrr", "evidence_ndcg_at_k", "latency_ms",
    "community_report_coverage",
}


def paired_bootstrap(
    report: dict[str, Any], *,
    candidate: str, baseline: str,
    metric: str, dataset: str | None = None,
    iterations: int = 2000, seed: int = 42,
) -> dict[str, Any]:
    """Two-sided 95% paired percentile CI on per-question effect.

    This is a within-dataset and within-case comparison; confidence intervals
    for mixed task types should be reported in separate datasets.
    """
    if candidate == baseline or metric not in ALLOWED_METRICS:
        raise ValueError("Use distinct conditions and a supported metric")
    if iterations < 100:
        raise ValueError("At least 100 bootstrap iterations required")
    pairs: dict[str, dict[str, float]] = {}
    seen: set[tuple[str, str]] = set()
    for row in report.get("results", []):
        if dataset is not None and row.get("dataset") != dataset:
            continue
        condition = row.get("condition")
        if condition not in {candidate, baseline}:
            continue
        key = (str(row.get("case_id")), condition)
        if key in seen:
            raise ValueError(f"Duplicate condition row: {key}")
        seen.add(key)
        value = row.get(metric)
        if not row.get("success") or not isinstance(value, (int, float)):
            continue
        if isinstance(value, bool) or not math.isfinite(value):
            continue
        pairs.setdefault(key[0], {})[condition] = float(value)
    differences = [
        values[candidate] - values[baseline]
        for _, values in sorted(pairs.items())
        if candidate in values and baseline in values
    ]
    if len(differences) < 2:
        raise ValueError("Need at least two paired, valid question outcomes")
    observed = mean(differences)
    rng = random.Random(seed)
    n = len(differences)
    bootstraps = sorted(
        mean(differences[rng.randrange(n)] for _ in range(n))
        for _ in range(iterations)
    )
    low = bootstraps[int(0.025 * (iterations - 1))]
    high = bootstraps[int(0.975 * (iterations - 1))]
    return {
        "candidate": candidate, "baseline": baseline,
        "metric": metric, "dataset": dataset,
        "n_paired": n,
        "candidate_minus_baseline": observed,
        "ci_95_percentile": [low, high],
        "wins": sum(diff > 0 for diff in differences),
        "ties": sum(diff == 0 for diff in differences),
        "losses": sum(diff < 0 for diff in differences),
        "seed": seed, "bootstrap_iterations": iterations,
        "interpretation": (
            "Negative difference means the candidate is faster"
            if metric == "latency_ms"
            else "Positive difference means higher candidate metric"
        ),
        "limitations": (
            "Questions are bootstrap units; use separate dataset strata. "
            "This percentile CI does not correct for model seed variance, "
            "shared papers, or judge bias."
        ),
    }


def benchmark_comparison_matrix(
    report: dict[str, Any], candidate: str, baseline: str,
    dataset: str | None = None,
) -> dict[str, Any]:
    """Collect safe, multi-axis comparisons while keeping NA explicit."""
    results = {}
    for metric in (
        "exact_match", "answer_token_f1", "rouge_l_f1",
        "evidence_recall", "evidence_mrr", "latency_ms"
    ):
        try:
            results[metric] = paired_bootstrap(
                report, candidate=candidate, baseline=baseline,
                dataset=dataset,
            )
        except ValueError as exc:
            results[metric] = {"available": False, "reason": str(exc)}
    return results
