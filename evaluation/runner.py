"""Run the same evaluation cases across Basic, Local, Global, and DRIFT."""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

from evaluation.metrics import (
    aggregate_results,
    answer_term_coverage,
    source_recall,
)
from evaluation.models import EvaluationCase, EvaluationResult
from models.schemas import QueryMode
from workflows.query_workflow import query_workflow


def load_cases(path: Path) -> list[EvaluationCase]:
    cases = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                cases.append(EvaluationCase.model_validate_json(line))
            except Exception as exc:
                raise ValueError(
                    f"Invalid evaluation case at line {line_number}: {exc}"
                ) from exc
    return cases


def extract_source_ids(mode: QueryMode, context: dict[str, Any]) -> list[str]:
    source_ids: list[str] = []
    if mode in {"basic", "local"}:
        source_ids.extend(
            item.get("id")
            for item in context.get("text_units", [])
            if item.get("id")
        )
    if mode == "local":
        source_ids.extend(
            f"entity:{item.get('name')}"
            for item in context.get("seed_entities", [])
            if item.get("name")
        )
        source_ids.extend(
            item.get("id")
            for item in context.get("claims", [])
            if item.get("id")
        )
    if mode in {"global", "drift"}:
        source_ids.extend(
            item.get("id")
            for item in context.get("community_summaries", [])
            if item.get("id")
        )
    if mode == "drift":
        drift = context.get("drift", {})
        for evidence in drift.get("evidence", []):
            source_ids.extend(evidence.get("source_ids", []))
    return list(dict.fromkeys(source_ids))


async def run_case(
    case: EvaluationCase,
    mode: QueryMode,
) -> EvaluationResult:
    started = perf_counter()
    final_state = await query_workflow.run(
        query=case.query,
        index_id=case.index_id,
        mode=mode,
        top_k=case.top_k,
        community_level=case.community_level,
    )
    latency_ms = (perf_counter() - started) * 1000.0
    error = final_state.get("error")
    context = final_state.get("relevant_context", {})
    answer = final_state.get("answer", "")
    source_ids = extract_source_ids(mode, context)

    drift_actions = None
    if mode == "drift":
        drift_actions = context.get("drift", {}).get("actions_executed")

    return EvaluationResult(
        case_id=case.id,
        mode=mode,
        success=not bool(error),
        answer=answer,
        source_ids=source_ids,
        source_recall=source_recall(
            case.expected_source_ids,
            source_ids,
        ),
        answer_term_coverage=answer_term_coverage(
            case.expected_answer_terms,
            answer,
        ),
        latency_ms=latency_ms,
        error=error,
        drift_actions=drift_actions,
    )


async def run_dataset(cases: list[EvaluationCase]) -> dict:
    results: list[EvaluationResult] = []
    for case in cases:
        for mode in case.modes:
            results.append(await run_case(case, mode))
    return {
        "summary": aggregate_results(results),
        "results": [item.model_dump() for item in results],
    }


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
