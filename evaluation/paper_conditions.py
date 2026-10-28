"""Six-condition experimental harness for Edge et al., arXiv:2404.16130.

Conditions C0-C3 use Global Search with complete hierarchical cuts.
TS is source text Map/Reduce. SS is basic semantic retrieval.
Do not confuse this runnable harness with a replication of paper results.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from evaluation.metrics import aggregate_results
from evaluation.models import EvaluationCase
from evaluation.runner import run_case


CONDITIONS = {
    "C0": ("global", 0),
    "C1": ("global", 1),
    "C2": ("global", 2),
    "C3": ("global", 3),
    "TS": ("text_summary", None),
    "SS": ("basic", None),
}


async def run_paper_conditions(
    cases: list[EvaluationCase],
    conditions: list[str] | None = None,
    max_cases: int = 125,
) -> dict[str, Any]:
    """Run each question against the requested conditions and retain labels.

    All conditions share the case's index, query, and model configuration.
    Downstream analysis must additionally verify matched generation budgets.
    """
    if max_cases <= 0:
        raise ValueError("max_cases must be positive")
    chosen = list(CONDITIONS) if conditions is None else list(dict.fromkeys(conditions))
    if not chosen:
        raise ValueError("At least one condition is required")
    invalid = [condition for condition in chosen if condition not in CONDITIONS]
    if invalid:
        raise ValueError(f"Unknown paper evaluation conditions: {invalid}")
    evaluated = cases[:max_cases]
    results = []
    grouped = defaultdict(list)
    for case in evaluated:
        for condition in chosen:
            mode, level = CONDITIONS[condition]
            scoped = case.model_copy(update={"community_level": level})
            item = await run_case(scoped, mode)
            grouped[condition].append(item)
            results.append({
                **item.model_dump(),
                "condition": condition,
                "community_level_requested": level,
            })
    summary = {}
    for condition in chosen:
        mode, _ = CONDITIONS[condition]
        rows = grouped[condition]
        summary[condition] = aggregate_results(rows).get(mode, {})
    return {
        "method": "six_condition_paper_harness",
        "conditions": chosen,
        "num_cases": len(evaluated),
        "questions": {case.id: case.query for case in evaluated},
        "summary": summary,
        "results": results,
        "limitations": [
            "Does not reproduce the original paper's datasets or GPT-4 Turbo setup.",
            "Paper-level result parity requires model, prompt, chunking, and budget audits.",
        ],
    }
