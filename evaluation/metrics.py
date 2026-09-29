"""Deterministic evaluation metrics that do not require an LLM judge."""
from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Iterable

from evaluation.models import EvaluationResult


def source_recall(expected: Iterable[str], actual: Iterable[str]) -> float:
    expected_set = {item for item in expected if item}
    if not expected_set:
        return 1.0
    actual_set = {item for item in actual if item}
    return len(expected_set & actual_set) / len(expected_set)


def answer_term_coverage(expected_terms: Iterable[str], answer: str) -> float:
    terms = [term.casefold().strip() for term in expected_terms if term.strip()]
    if not terms:
        return 1.0
    normalized_answer = answer.casefold()
    matched = sum(1 for term in terms if term in normalized_answer)
    return matched / len(terms)


def aggregate_results(results: list[EvaluationResult]) -> dict:
    grouped = defaultdict(list)
    for result in results:
        grouped[result.mode].append(result)

    summary = {}
    for mode, items in sorted(grouped.items()):
        summary[mode] = {
            "cases": len(items),
            "success_rate": mean(1.0 if item.success else 0.0 for item in items),
            "avg_source_recall": mean(item.source_recall for item in items),
            "avg_answer_term_coverage": mean(
                item.answer_term_coverage for item in items
            ),
            "avg_latency_ms": mean(item.latency_ms for item in items),
            "avg_drift_actions": (
                mean(
                    item.drift_actions
                    for item in items
                    if item.drift_actions is not None
                )
                if any(item.drift_actions is not None for item in items)
                else None
            ),
        }
    return summary
