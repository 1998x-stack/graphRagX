"""Extraction correctness: invalid LLM responses cannot create a false-success index."""
import pytest

from core.extraction import EntityRelationExtractor
from models.schemas import TextChunk


@pytest.mark.asyncio
async def test_extraction_retries_exactly_bounded_and_raises(monkeypatch):
    calls = []

    async def invalid_generate(*args, **kwargs):
        calls.append(1)
        return "the model did not return JSON"

    monkeypatch.setattr("core.extraction.llm_service.generate", invalid_generate)
    extractor = EntityRelationExtractor()
    extractor.max_retry = 2
    chunk = TextChunk(id="doc_chunk_0", text="Acme bought Beta.", doc_id="doc", chunk_index=0)

    with pytest.raises(RuntimeError, match="Extraction failed for chunk"):
        await extractor.extract_from_chunk(chunk)
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_extraction_distinguishes_real_empty_results(monkeypatch):
    async def valid_empty(*args, **kwargs):
        return '{"entities":[],"relations":[]}'

    monkeypatch.setattr("core.extraction.llm_service.generate", valid_empty)
    extractor = EntityRelationExtractor()
    chunk = TextChunk(id="x_0", text="No named entities.", doc_id="x", chunk_index=0)
    result = await extractor.extract_from_chunk(chunk)
    assert result.chunk_id == chunk.id
    assert result.entities == []
    assert result.relations == []


@pytest.mark.asyncio
async def test_extraction_rejects_relations_with_unknown_endpoints(monkeypatch):
    async def dangling(*args, **kwargs):
        return (
            '{"entities":[{"name":"A","type":"CONCEPT","description":"A"}],'
            '"relations":[{"source":"A","target":"Missing",'
            '"relation_type":"RELATED","description":"invalid"}]}'
        )

    monkeypatch.setattr("core.extraction.llm_service.generate", dangling)
    extractor = EntityRelationExtractor()
    extractor.max_retry = 0
    chunk = TextChunk(id="x_0", text="A", doc_id="x", chunk_index=0)
    with pytest.raises(RuntimeError, match="Extraction failed"):
        await extractor.extract_from_chunk(chunk)
