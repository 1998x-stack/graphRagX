"""Deterministic character/token text units with auditable source offsets."""
from __future__ import annotations

from typing import List, Literal

from config import settings
from models.schemas import TextChunk
from utils.logger import log
from utils.token_budget import token_budget


ChunkUnit = Literal["characters", "tokens"]


class TextChunker:
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        unit: ChunkUnit | None = None,
    ):
        self.chunk_size = settings.CHUNK_SIZE if chunk_size is None else chunk_size
        self.chunk_overlap = settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
        self.unit: ChunkUnit = settings.CHUNK_UNIT if unit is None else unit
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")
        if self.unit not in {"characters", "tokens"}:
            raise ValueError("unit must be 'characters' or 'tokens'")

    @staticmethod
    def _text_chunk(
        text: str,
        doc_id: str,
        index: int,
        start_char: int,
        end_char: int,
        metadata: dict | None,
        start_token: int | None = None,
        end_token: int | None = None,
    ) -> TextChunk | None:
        original = text[start_char:end_char]
        trim_left = len(original) - len(original.lstrip())
        trim_right = len(original) - len(original.rstrip())
        start_char += trim_left
        end_char -= trim_right
        if start_char >= end_char:
            return None
        # The exact source substring is retained: no lossy encode/decode splice.
        return TextChunk(
            id=f"{doc_id}_chunk_{index}",
            text=text[start_char:end_char],
            doc_id=doc_id,
            chunk_index=index,
            metadata=dict(metadata or {}),
            start_char=start_char,
            end_char=end_char,
            start_token=start_token,
            end_token=end_token,
        )

    def _chunk_characters(
        self, text: str, doc_id: str, metadata: dict | None
    ) -> List[TextChunk]:
        chunks: List[TextChunk] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            candidate = text[start:end]
            if end < len(text):
                boundary = max(candidate.rfind(mark) for mark in ".?!。？！\n")
                if boundary >= self.chunk_size // 2:
                    end = start + boundary + 1
            chunk = self._text_chunk(
                text, doc_id, len(chunks), start, end, metadata
            )
            if chunk is not None:
                chunks.append(chunk)
            if end >= len(text):
                break
            next_start = end - self.chunk_overlap
            if next_start <= start:
                raise RuntimeError("chunker failed to make forward progress")
            start = next_start
        return chunks

    def _chunk_tokens(
        self, text: str, doc_id: str, metadata: dict | None
    ) -> List[TextChunk]:
        encoding = token_budget.encoding
        ids = encoding.encode(text, disallowed_special=())
        if not ids:
            return []
        decoded, offsets = encoding.decode_with_offsets(ids)
        if decoded != text or len(offsets) != len(ids):
            raise ValueError("Tokenizer did not preserve source text offsets")
        chunks: List[TextChunk] = []
        token_start = 0
        while token_start < len(ids):
            token_end = min(token_start + self.chunk_size, len(ids))
            start_char = offsets[token_start]
            end_char = offsets[token_end] if token_end < len(ids) else len(text)
            chunk = self._text_chunk(
                text, doc_id, len(chunks), start_char, end_char, metadata,
                start_token=token_start, end_token=token_end,
            )
            if chunk is not None:
                chunks.append(chunk)
            if token_end >= len(ids):
                break
            token_start = token_end - self.chunk_overlap
        return chunks

    def chunk_text(
        self, text: str, doc_id: str, metadata: dict | None = None
    ) -> List[TextChunk]:
        if not text or not text.strip():
            return []
        if self.unit == "tokens":
            chunks = self._chunk_tokens(text, doc_id, metadata)
        else:
            chunks = self._chunk_characters(text, doc_id, metadata)
        log.debug(
            "Chunked doc {} into {} {} chunks", doc_id, len(chunks), self.unit
        )
        return chunks

    def chunk_documents(
        self, documents: List[str], doc_ids: List[str] | None = None
    ) -> List[TextChunk]:
        doc_ids = doc_ids or [f"doc_{i}" for i in range(len(documents))]
        if len(documents) != len(doc_ids):
            raise ValueError("documents and doc_ids must have the same length")
        chunks: List[TextChunk] = []
        for document, doc_id in zip(documents, doc_ids):
            chunks.extend(self.chunk_text(document, doc_id))
        return chunks


text_chunker = TextChunker()
