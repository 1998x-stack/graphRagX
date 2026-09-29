import numpy as np
import pytest

from services.embedding_service import HashEmbeddingService


@pytest.mark.asyncio
async def test_hash_embeddings_are_deterministic():
    service = HashEmbeddingService(dimension=128)
    first = await service.embed_text("graph retrieval graph")
    second = await service.embed_text("graph retrieval graph")
    assert np.array_equal(first, second)


@pytest.mark.asyncio
async def test_hash_embedding_ranking_has_lexical_signal():
    service = HashEmbeddingService(dimension=256)
    query = await service.embed_text("knowledge graph retrieval")
    candidates = await service.embed_texts(
        ["knowledge graph retrieval system", "banana recipe kitchen", "weather forecast ocean"]
    )
    assert service.find_most_similar(query, candidates, top_k=1) == [0]
