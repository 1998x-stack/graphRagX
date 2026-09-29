"""Embedding providers and vector similarity helpers."""
from __future__ import annotations

import asyncio
import hashlib
import re
from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np

from config import settings
from utils.logger import log, log_exception


class EmbeddingService(ABC):
    def __init__(self, model: str | None = None, dimension: int | None = None):
        self.model = model or settings.EMBEDDING_MODEL
        self.dimension = dimension or settings.EMBEDDING_DIMENSION

    @abstractmethod
    async def embed_texts(self, texts: Sequence[str]) -> List[np.ndarray]:
        raise NotImplementedError

    async def embed_text(self, text: str) -> np.ndarray:
        vectors = await self.embed_texts([text])
        return vectors[0]

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        denominator = float(np.linalg.norm(vec1) * np.linalg.norm(vec2))
        if denominator == 0.0:
            return 0.0
        return float(np.dot(vec1, vec2) / denominator)

    def find_most_similar(
        self,
        query_embedding: np.ndarray,
        embeddings: Sequence[np.ndarray],
        top_k: int = 5,
    ) -> List[int]:
        if not embeddings or top_k <= 0:
            return []
        similarities = np.asarray(
            [self.cosine_similarity(query_embedding, item) for item in embeddings],
            dtype=np.float32,
        )
        k = min(top_k, len(embeddings))
        return np.argsort(similarities)[::-1][:k].tolist()


class HashEmbeddingService(EmbeddingService):
    """Deterministic, offline lexical embeddings for development and tests.

    This is intentionally not presented as a semantic embedding model. It is a
    stable feature-hashing baseline that makes the default project reproducible
    and useful without credentials.
    """

    TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = self.TOKEN_RE.findall(text.casefold())
        if not tokens:
            return vector

        features = tokens + [f"{a}::{b}" for a, b in zip(tokens, tokens[1:])]
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimension
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign

        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector

    async def embed_texts(self, texts: Sequence[str]) -> List[np.ndarray]:
        return [self._embed(text) for text in texts]


class OpenAIEmbeddingService(EmbeddingService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not settings.EMBEDDING_API_KEY:
            raise ValueError("EMBEDDING_API_KEY is required when EMBEDDING_PROVIDER=openai")
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(
            api_key=settings.EMBEDDING_API_KEY,
            base_url=settings.EMBEDDING_BASE_URL or None,
        )

    async def embed_texts(self, texts: Sequence[str]) -> List[np.ndarray]:
        if not texts:
            return []

        vectors: List[np.ndarray] = []
        batch_size = settings.EMBEDDING_BATCH_SIZE
        for offset in range(0, len(texts), batch_size):
            batch = list(texts[offset : offset + batch_size])
            for attempt in range(settings.MAX_RETRY + 1):
                try:
                    response = await self.client.embeddings.create(
                        model=self.model,
                        input=batch,
                        dimensions=self.dimension,
                    )
                    vectors.extend(
                        np.asarray(item.embedding, dtype=np.float32)
                        for item in response.data
                    )
                    break
                except Exception as exc:
                    if attempt >= settings.MAX_RETRY:
                        log_exception(exc, "OpenAIEmbeddingService.embed_texts")
                        raise
                    await asyncio.sleep(settings.RETRY_DELAY * (2**attempt))
        return vectors


def create_embedding_service(provider: str | None = None) -> EmbeddingService:
    selected = provider or settings.EMBEDDING_PROVIDER
    if selected == "hash":
        return HashEmbeddingService()
    if selected == "openai":
        return OpenAIEmbeddingService()
    raise ValueError(f"Unsupported embedding provider: {selected}")


embedding_service = create_embedding_service()
log.info(
    "Embedding service initialized: provider={}, model={}, dimension={}",
    settings.EMBEDDING_PROVIDER,
    embedding_service.model,
    embedding_service.dimension,
)
