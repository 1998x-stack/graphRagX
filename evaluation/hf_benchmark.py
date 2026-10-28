"""One frozen index per QA context; paired retrieval modes share source data.

Reports distinguish oracle-context tests from true open-corpus retrieval.
No upstream network call is made here; HF download is a separate step.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from config import settings
from evaluation.hf_datasets import BenchmarkCase
from evaluation.hf_metrics import aggregate_slices, answer_metrics, evidence_metrics
from evaluation.paper_conditions import CONDITIONS
from models.schemas import GraphData
from services.storage_service import storage_service
from workflows.indexing_workflow import indexing_workflow
from workflows.query_workflow import query_workflow


def index_id_for_case(case: BenchmarkCase) -> str:
    identity = f"{case.dataset}:{case.revision}:{case.id}"
    return "hf_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _extract_candidate_docs(
    context: dict, mode: str, graph: GraphData, case: BenchmarkCase,
    index_id: str,
) -> list[str]:
    """Map retrieval candidates to original document IDs.

    For Global, expansion through community entities is *potential evidence
    exposure*, not verified source support for the generated answer.
    """
    document_lookup = {
        f"{index_id}_doc_{i}": document.id
        for i, document in enumerate(case.documents)
    }
    chunk_lookup = {chunk.id: chunk.doc_id for chunk in graph.text_chunks}
    collected: list[str] = []

    def include_chunk(chunk_id: str) -> None:
        doc_id = document_lookup.get(chunk_lookup.get(chunk_id, ""))
        if doc_id:
            collected.append(doc_id)

    if mode in {"basic", "local"}:
        for text_unit in context.get("text_units", []):
            document_id = document_lookup.get(text_unit.get("doc_id", ""))
            if document_id:
                collected.append(document_id)
    elif mode == "text_summary":
        for text_unit in context.get("community_summaries", []):
            include_chunk(text_unit.get("id", ""))
    elif mode == "global":
        # Prefer map points that actually survived score filtering.
        referenced = [
            source
            for point in context.get("map_points", [])
            for source in point.get("source_reports", [])
        ]
        if not referenced:
            referenced = [
                item.get("id")
                for item in context.get("community_summaries", [])
            ]
        by_id = {community.id: community for community in graph.communities}
        for community_id in referenced:
            community = by_id.get(community_id)
            if not community:
                continue
            for entity_name in community.entities:
                entity = graph.entities.get(entity_name)
                if entity:
                    for chunk_id in entity.source_chunk_ids:
                        include_chunk(chunk_id)
    elif mode == "drift":
        for item in context.get("drift", {}).get("evidence", []):
            for source in item.get("source_ids", []):
                include_chunk(source)
    return list(dict.fromkeys(collected))


def _condition(mode_or_condition: str) -> tuple[str, int | None]:
    if mode_or_condition in CONDITIONS:
        return CONDITIONS[mode_or_condition]
    if mode_or_condition in {"local", "global", "basic", "drift", "text_summary"}:
        return mode_or_condition, None
    raise ValueError(f"Unknown benchmark condition: {mode_or_condition}")


async def run_benchmark(
    cases: list[BenchmarkCase],
    conditions: list[str],
    *,
    allow_mock_smoke: bool = False,
    overwrite: bool = False,
    k: int = 5,
) -> dict[str, Any]:
    """Build each index once, then query all conditions on that frozen index.

    We deliberately do not label the mock/hash setup as a performance study.
    Query failures and index failures remain visible in every report row.
    """
    if not cases or not conditions or k < 1:
        raise ValueError("Nonempty cases/conditions and positive k are required")
    for condition in conditions:
        _condition(condition)
    offline = settings.LLM_PROVIDER == "mock" or settings.EMBEDDING_PROVIDER == "hash"
    if offline and not allow_mock_smoke:
        raise RuntimeError(
            "Real benchmark requires real LLM and embedding providers. "
            "Pass --allow-mock-smoke only for correctness smoke tests."
        )
    records: list[dict[str, Any]] = []
    index_records = []
    for case in cases:
        index_id = index_id_for_case(case)
        documents = [doc.text for doc in case.documents]
        if len(documents) > settings.MAX_DOCUMENTS:
            raise ValueError(f"{case.id} has too many documents")
        if sum(len(doc) for doc in documents) > settings.MAX_DOCUMENT_CHARS:
            raise ValueError(f"{case.id} exceeds MAX_DOCUMENT_CHARS")
        started = perf_counter()
        index_error = None
        if storage_service.index_exists(index_id) and not overwrite:
            graph = storage_service.load_graph(index_id)
            recorded = graph.metadata.get("source_metadata", {})
            if recorded.get("hf_case_id") != case.id or recorded.get("hf_revision") != case.revision:
                raise ValueError(f"Existing index {index_id} is not this dataset snapshot")
            # Reuse is safe only with identical content and model parameters.
            metadata = graph.metadata
            if (
                metadata.get("llm_provider") != settings.LLM_PROVIDER
                or metadata.get("llm_model") != settings.LLM_MODEL
                or metadata.get("embedding_provider") != settings.EMBEDDING_PROVIDER
                or metadata.get("embedding_model") != settings.EMBEDDING_MODEL
                or metadata.get("community_algorithm") != settings.COMMUNITY_ALGORITHM
                or metadata.get("chunking", {}).get("unit") != settings.CHUNK_UNIT
                or metadata.get("chunking", {}).get("size") != settings.CHUNK_SIZE
                or metadata.get("chunking", {}).get("overlap") != settings.CHUNK_OVERLAP
            ):
                raise ValueError("Existing index model/algorithm/chunking configuration differs")
            hashes = metadata.get("input_document_sha256", {})
            for i, doc in enumerate(documents):
                digest = hashlib.sha256(doc.encode("utf-8")).hexdigest()
                if hashes.get(f"{index_id}_doc_{i}") != digest:
                    raise ValueError("Existing index source hash mismatch")
            indexed = {"current_step": "reused"}
        else:
            indexed = await indexing_workflow.run(
                index_id, documents,
                metadata={
                    "hf_case_id": case.id,
                    "hf_dataset": case.dataset,
                    "hf_revision": case.revision,
                    "hf_split": case.split,
                    "source_documents": [item.model_dump() for item in case.documents],
                },
            )
            index_error = indexed.get("error")
            graph = indexed.get("graph_data")
        index_ms = (perf_counter() - started) * 1000
        if not index_error and not storage_service.index_exists(index_id):
            index_error = "Index workflow reported success without persisted index"
        stats = {
            "case_id": case.id,
            "index_id": index_id,
            "index_ms": index_ms,
            "index_reused": indexed["current_step"] == "reused",
            "robustness_variant": case.metadata.get("robustness_variant", "original"),
            "index_error": index_error,
            "num_docs": len(documents),
            "num_chunks": len(graph.text_chunks) if graph else None,
            "num_entities": len(graph.entities) if graph else None,
            "num_relations": len(graph.relations) if graph else None,
            "num_communities": len(graph.communities) if graph else None,
        }
        index_records.append(stats)
        for condition in conditions:
            mode, level = _condition(condition)
            started_query = perf_counter()
            if index_error:
                result = {"error": index_error, "answer": "", "relevant_context": {}}
            else:
                result = await query_workflow.run(
                    query=case.question, index_id=index_id,
                    mode=mode, top_k=k, community_level=level
                )
            query_ms = (perf_counter() - started_query) * 1000
            context = result.get("relevant_context") or {}
            answer = result.get("answer") or ""
            success = not bool(result.get("error"))
            candidates = (
                _extract_candidate_docs(context, mode, graph, case, index_id)
                if success else []
            )
            # Gold source annotations are available for HotpotQA and MuSiQue.
            evidence = evidence_metrics(
                case.gold_document_ids, candidates,
                annotated=bool(case.gold_document_ids), k=k,
            )
            row = {
                "case_id": case.id,
                "dataset": case.dataset,
                "task_type": case.task_type,
                "difficulty": case.metadata.get("difficulty"),
                "hop_count": case.metadata.get("hop_count"),
                "question_type": case.metadata.get("question_type"),
                "base_case_id": case.metadata.get("base_case_id", case.id),
                "robustness_variant": case.metadata.get("robustness_variant", "original"),
                "robustness_seed": case.metadata.get("robustness_seed"),
                "condition": condition,
                "mode": mode,
                "community_level": level,
                "success": success,
                "error": result.get("error"),
                "answer": answer,
                "reference_answers": case.reference_answers,
                "retrieved_document_ids": candidates,
                "gold_document_ids": case.gold_document_ids,
                "index_id": index_id,
                "latency_ms": query_ms,
                "index_ms": index_ms,
                "community_report_coverage": context.get("community_report_coverage"),
                "community_cut_size": context.get("community_cut_size"),
                "community_reports_included": context.get("community_reports_included"),
                "drift_actions": context.get("drift", {}).get("actions_executed"),
                **(
                    answer_metrics(answer, case.reference_answers, case.task_type)
                    if success else {
                        "exact_match": None,
                        "answer_token_f1": None,
                        "rouge_l_f1": None,
                    }
                ),
                **evidence,
            }
            records.append(row)
    return {
        "benchmark_schema_version": "1.0",
        "mode": "offline_smoke" if offline else "real_model_evaluation",
        "measurement_warnings": [
            "Source document recall measures candidate exposure, not citation faithfulness.",
            "The QA corpus is an oracle candidate set, not open-world Wikipedia retrieval.",
            "Global and TS expose broader contexts; retrieval cost and budgets must be matched.",
            "Token usage and monetary costs are not instrumented: no cost estimates reported.",
            "No factuality, human preference or answer grounding score without annotation or judge.",
        ],
        "settings": {
            "llm_provider": settings.LLM_PROVIDER,
            "llm_model": settings.LLM_MODEL,
            "embedding_provider": settings.EMBEDDING_PROVIDER,
            "embedding_model": settings.EMBEDDING_MODEL,
            "chunk_unit": settings.CHUNK_UNIT,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "community_algorithm": settings.COMMUNITY_ALGORITHM,
            "global_selection": settings.GLOBAL_SELECTION_MODE,
            "top_k": k,
        },
        "index_records": index_records,
        "questions": {case.id: case.question for case in cases},
        "summary": aggregate_slices(records),
        "results": records,
    }


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
