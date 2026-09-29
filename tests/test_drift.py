import pytest

import workflows.query_workflow as query_module
from config import settings
from models.graph import KnowledgeGraph
from models.schemas import Community, Entity, TextChunk
from workflows.query_workflow import QueryWorkflow


def build_drift_fixture():
    kg = KnowledgeGraph()
    entity = Entity(
        name="GraphRAG",
        type="CONCEPT",
        description="Graph retrieval system",
        source_chunk_ids=["chunk_0"],
    )
    kg.add_entity(entity)
    community = Community(
        id="L0_C0",
        level=0,
        entities=["GraphRAG"],
        summary="GraphRAG combines graph and retrieval methods.",
        size=1,
    )
    kg.set_communities([community])
    graph_data = kg.to_graph_data()
    graph_data.text_chunks = [
        TextChunk(
            id="chunk_0",
            text="GraphRAG connects community summaries with local graph evidence.",
            doc_id="doc_0",
            chunk_index=0,
        )
    ]
    return kg, graph_data


@pytest.mark.asyncio
async def test_drift_offline_primer_followup_reduce():
    workflow = QueryWorkflow()
    kg, graph_data = build_drift_fixture()

    explored = await workflow.drift_search(
        {
            "query": "How does GraphRAG combine broad and local evidence?",
            "index_id": "demo",
            "mode": "drift",
            "top_k": 5,
            "community_level": 0,
            "kg_object": kg,
            "graph_data": graph_data,
        }
    )

    assert explored["current_step"] == "drift_explored"
    trace = explored["drift_trace"]
    assert trace.primer_answer
    assert trace.actions_executed == 1
    assert trace.max_depth_reached == 1
    assert len(trace.evidence) == 1
    assert "chunk_0" in trace.evidence[0].source_ids

    reduced = await workflow.generate_drift_answer(
        {
            "query": "How does GraphRAG combine broad and local evidence?",
            "drift_trace": trace,
            "relevant_context": explored["relevant_context"],
        }
    )
    assert reduced["current_step"] == "completed"
    assert reduced["answer"]


@pytest.mark.asyncio
async def test_drift_low_confidence_stops_expansion(monkeypatch):
    workflow = QueryWorkflow()
    kg, graph_data = build_drift_fixture()

    async def fake_generate(prompt, task="general", **kwargs):
        del prompt, kwargs
        if task == "drift_primer":
            return (
                '{"answer":"primer","follow_ups":['
                '{"question":"Inspect GraphRAG locally","score":90}]}'
            )
        if task.startswith("drift_followup_"):
            return (
                '{"answer":"weak evidence","confidence":0.1,"follow_ups":['
                '{"question":"This should not execute","score":99}]}'
            )
        if task == "drift_reduce":
            return "reduced"
        return "{}"

    monkeypatch.setattr(query_module.llm_service, "generate", fake_generate)
    monkeypatch.setattr(settings, "DRIFT_N_DEPTH", 3)
    monkeypatch.setattr(settings, "DRIFT_K_FOLLOWUPS", 2)
    monkeypatch.setattr(settings, "DRIFT_EXPANSION_MIN_CONFIDENCE", 0.5)

    explored = await workflow.drift_search(
        {
            "query": "Explain GraphRAG",
            "index_id": "demo",
            "mode": "drift",
            "top_k": 5,
            "community_level": 0,
            "kg_object": kg,
            "graph_data": graph_data,
        }
    )

    trace = explored["drift_trace"]
    assert trace.actions_executed == 1
    assert trace.max_depth_reached == 1
    assert trace.termination_reason == "no_followups"
    assert [item.question for item in trace.evidence] == [
        "Inspect GraphRAG locally"
    ]


def test_drift_followup_parser_clamps_values():
    answer, confidence, followups = QueryWorkflow._parse_drift_followup(
        (
            '{"answer":"evidence","confidence":5,"follow_ups":['
            '{"question":"next","score":200}]}'
        ),
        depth=1,
        parent_id="q0",
    )
    assert answer == "evidence"
    assert confidence == 1.0
    assert followups[0].score == 100.0
    assert followups[0].depth == 2
    assert followups[0].parent_id == "q0"
