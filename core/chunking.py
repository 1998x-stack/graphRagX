"""Deterministic text chunking with overlap and sentence-boundary alignment."""
from typing import List

from config import settings
from models.schemas import TextChunk
from utils.logger import log


class TextChunker:
    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")

    def chunk_text(self, text: str, doc_id: str, metadata: dict | None = None) -> List[TextChunk]:
        if not text or not text.strip():
            return []

        chunks: List[TextChunk] = []
        start = 0
        index = 0
        length = len(text)
        while start < length:
            end = min(start + self.chunk_size, length)
            candidate = text[start:end]

            if end < length:
                boundaries = [candidate.rfind(mark) for mark in ".?!。？！\n"]
                boundary = max(boundaries)
                if boundary >= self.chunk_size // 2:
                    end = start + boundary + 1
                    candidate = text[start:end]

            cleaned = candidate.strip()
            if cleaned:
                chunks.append(
                    TextChunk(
                        id=f"{doc_id}_chunk_{index}",
                        text=cleaned,
                        doc_id=doc_id,
                        chunk_index=index,
                        metadata=metadata or {},
                    )
                )
                index += 1

            if end >= length:
                break
            next_start = end - self.chunk_overlap
            if next_start <= start:
                raise RuntimeError("chunker failed to make forward progress")
            start = next_start

        log.debug("Chunked doc {} into {} chunks", doc_id, len(chunks))
        return chunks

    def chunk_documents(self, documents: List[str], doc_ids: List[str] | None = None) -> List[TextChunk]:
        doc_ids = doc_ids or [f"doc_{i}" for i in range(len(documents))]
        if len(documents) != len(doc_ids):
            raise ValueError("documents and doc_ids must have the same length")
        chunks: List[TextChunk] = []
        for document, doc_id in zip(documents, doc_ids):
            chunks.extend(self.chunk_text(document, doc_id))
        return chunks


text_chunker = TextChunker()
