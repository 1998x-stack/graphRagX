"""Original paper's Text Summary (TS) control-arm regression tests."""
import pytest

from config import settings
from models.schemas import GraphData, TextChunk
from workflows.query_workflow import QueryWorkflow


def _text_units():
    return [
        TextChunk(
            id=f"doc_{i}_chunk_0",
            doc_id=f"doc_{i}",
            chunk_index=0,
            text=f"Independent source document {i} on theme {i}. " * 8,
        )
        for i in range(4)
    ]


@pytest.mark.asyncio
async def test_text_summary_covers_all_units_and_runs_map_reduce():
    workflow = QueryWorkflow()
    state = {
        "query": "Which themes recur?",
        "index_id": "unit-test",
        "mode": "text_summary",
        "top_k": 1,
        "graph_data": GraphData(text_chunks=_text_units()),
    }
    retrieved = await workflow.text_summary_search(state)
    assert not retrieved.get("error")
    context = retrieved["relevant_context"]
    assert context["community_reports_included"] == 4
    assert context["community_report_coverage"] == 1.0
    assert {item["id"] for item in context["community_summaries"]} == {
        item.id for item in _text_units()
    }
    answered = await workflow.generate_global_answer({
        **state,
        "relevant_context": context,
    })
    assert answered["current_step"] == "completed"
    assert answered["answer"]


@pytest.mark.asyncio
async def test_text_summary_guard_is_explicit(monkeypatch):
    monkeypatch.setattr(settings, "TS_MAX_TEXT_UNITS", 2)
    state = {
        "query": "All themes",
        "graph_data": GraphData(text_chunks=_text_units()),
    }
    result = await QueryWorkflow().text_summary_search(state)
    assert "TS_MAX_TEXT_UNITS" in result["error"]
