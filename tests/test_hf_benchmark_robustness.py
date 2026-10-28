"""Full benchmark control flow with synthetic one-index-per-case fixtures."""
import hashlib

import pytest

from config import settings
from evaluation.hf_benchmark import index_id_for_case, run_benchmark
from evaluation.hf_datasets import adapt_row
from evaluation.hf_robustness import expand_variants, make_variant
from models.schemas import Community, Entity, GraphData, TextChunk
from services.storage_service import storage_service


REV = "abcdef1234567890"


def _case():
    return adapt_row("hotpotqa", {
        "id": "fixture", "question": "Who made the machine?",
        "answer": "Z", "level": "medium", "type": "bridge",
        "context": {
            "title": ["A", "B", "Noise"],
            "sentences": [["Machine made by Z."], ["Z works for A."], ["Irrelevant."]],
        },
        "supporting_facts": {
            "title": ["A", "B"], "sent_id": [0, 0],
        },
    }, REV, "validation")[0]


def test_robustness_preserves_gold_identity_and_does_not_invent_abstention():
    case = _case()
    cases = expand_variants(
        [case], ["original", "shuffle", "distractor", "drop_support"], seed=8
    )
    assert len(cases) == 4
    for item in cases[:3]:
        assert sorted(item.gold_document_ids) == ["d0", "d1"]
        assert item.reference_answers == ["Z"]
        assert item.metadata["base_case_id"] == case.id
    shuffled = cases[1]
    assert set(doc.id for doc in shuffled.documents) == {
        doc.id for doc in case.documents
    }
    noisy = cases[2]
    assert len(noisy.documents) == len(case.documents) + 2
    removed = cases[3]
    assert removed.reference_answers == []
    assert removed.gold_document_ids == []
    assert removed.metadata["answerability_after_removal"] == "unknown"
    assert [item.id for item in removed.documents] == ["d2"]
    with pytest.raises(ValueError, match="requires positive annotated gold"):
        make_variant(removed, "drop_support")


@pytest.mark.asyncio
async def test_benchmark_one_index_for_two_paired_conditions(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "hash")
    monkeypatch.setattr(storage_service, "graphs_dir", tmp_path)
    calls = []
    case = _case()

    async def fake_index(index_id, documents, metadata):
        calls.append(index_id)
        chunks = [
            TextChunk(id=f"{index_id}_doc_{i}_chunk_0", text=doc,
                      doc_id=f"{index_id}_doc_{i}", chunk_index=0)
            for i, doc in enumerate(documents)
        ]
        entities = {
            "Z": Entity(
                name="Z", type="PERSON", description="A worker",
                source_chunk_ids=[chunks[0].id, chunks[1].id],
            ),
        }
        graph = GraphData(
            text_chunks=chunks, entities=entities,
            communities=[Community(
                id="C0", level=0, entities=["Z"], summary="Z makes machines"
            )],
            metadata={
                "source_metadata": metadata,
                "llm_provider": settings.LLM_PROVIDER,
                "llm_model": settings.LLM_MODEL,
                "embedding_provider": settings.EMBEDDING_PROVIDER,
                "embedding_model": settings.EMBEDDING_MODEL,
                "community_algorithm": settings.COMMUNITY_ALGORITHM,
                "chunking": {
                    "unit": settings.CHUNK_UNIT,
                    "size": settings.CHUNK_SIZE,
                    "overlap": settings.CHUNK_OVERLAP,
                },
                "input_document_sha256": {
                    f"{index_id}_doc_{i}": hashlib.sha256(
                        doc.encode("utf-8")
                    ).hexdigest() for i, doc in enumerate(documents)
                },
            },
        )
        storage_service.save_graph(index_id, graph)
        return {"current_step": "completed", "graph_data": graph}

    async def fake_query(*, query, index_id, mode, top_k, community_level):
        graph = storage_service.load_graph(index_id)
        if mode == "basic":
            context = {
                "text_units": [item.model_dump()
                               for item in graph.text_chunks[:2]],
            }
        else:
            context = {
                "community_summaries": [{"id": "C0"}],
                "map_points": [{"source_reports": ["C0"]}],
                "community_report_coverage": 1.0,
            }
        return {"answer": "Z", "relevant_context": context, "error": None}

    monkeypatch.setattr("evaluation.hf_benchmark.indexing_workflow.run", fake_index)
    monkeypatch.setattr("evaluation.hf_benchmark.query_workflow.run", fake_query)

    report = await run_benchmark(
        [case], ["basic", "global"], allow_mock_smoke=True
    )
    assert calls == [index_id_for_case(case)]
    assert report["mode"] == "offline_smoke"
    assert len(report["results"]) == 2
    assert [row["condition"] for row in report["results"]] == ["basic", "global"]
    assert all(row["exact_match"] == 1.0 for row in report["results"])
    assert all(row["evidence_recall"] == 1.0 for row in report["results"])
    assert report["results"][1]["community_report_coverage"] == 1.0
    assert report["summary"]["condition"]["global"]["count"] == 1

    reused = await run_benchmark(
        [case], ["basic"], allow_mock_smoke=True
    )
    assert len(calls) == 1
    assert reused["index_records"][0]["index_reused"] is True


@pytest.mark.asyncio
async def test_mock_evaluation_needs_explicit_smoke_opt_in(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")
    with pytest.raises(RuntimeError, match="real LLM"):
        await run_benchmark([_case()], ["basic"])


@pytest.mark.asyncio
async def test_bad_condition_fails_before_indexing():
    with pytest.raises(ValueError, match="Unknown benchmark condition"):
        await run_benchmark([_case()], ["not_a_mode"], allow_mock_smoke=True)
