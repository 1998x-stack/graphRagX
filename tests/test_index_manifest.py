"""Reproducibility manifest and API contract tests for the v2.4 index."""
import hashlib

import pytest

from config import settings
from models.graph import KnowledgeGraph
from models.schemas import Entity, TextChunk
from workflows.indexing_workflow import IndexingWorkflow


@pytest.mark.asyncio
async def test_index_build_persists_document_hashes_and_chunk_model_settings():
    kg = KnowledgeGraph()
    kg.add_entity(Entity(
        name="Motor", type="DEVICE", description="A monitored electric motor",
        source_chunk_ids=["sample_doc_0_chunk_0"],
    ))
    document = "Motor vibration rises with bearing wear."
    chunk = TextChunk(
        id="sample_doc_0_chunk_0", doc_id="sample_doc_0",
        chunk_index=0, text=document, start_char=0, end_char=len(document),
    )
    workflow = IndexingWorkflow()
    result = await workflow.build_retrieval_index({
        "index_id": "sample",
        "documents": [document],
        "chunks": [chunk],
        "kg_object": kg,
        "claims": [],
        "metadata": {"study": "paper_fixture"},
    })
    assert result["current_step"] == "retrieval_index_built", result.get("error")
    index = result["graph_data"]
    meta = index.metadata
    assert meta["schema_version"] == settings.INDEX_SCHEMA_VERSION
    assert meta["llm_model"] == settings.LLM_MODEL
    assert meta["chunking"]["unit"] == settings.CHUNK_UNIT
    assert meta["chunking"]["size"] == settings.CHUNK_SIZE
    assert meta["chunking"]["overlap"] == settings.CHUNK_OVERLAP
    assert meta["input_document_sha256"]["sample_doc_0"] == hashlib.sha256(
        document.encode("utf-8")
    ).hexdigest()
    assert len(index.chunk_embeddings[chunk.id]) == settings.EMBEDDING_DIMENSION
    assert meta["source_metadata"]["study"] == "paper_fixture"
