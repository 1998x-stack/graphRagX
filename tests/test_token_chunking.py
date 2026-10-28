"""Paper 600/100 token window and exact provenance tests."""
import pytest

from core.chunking import TextChunker
from models.schemas import TextChunk


def test_token_chunking_preserves_source_offsets_and_overlap():
    text = "GraphRAG uses document evidence. " * 30
    chunker = TextChunker(chunk_size=20, chunk_overlap=4, unit="tokens")
    chunks = chunker.chunk_text(text, "paper_doc", metadata={"source": "fixture"})
    assert len(chunks) > 2
    for index, chunk in enumerate(chunks):
        assert chunk.text == text[chunk.start_char:chunk.end_char]
        assert chunk.metadata["source"] == "fixture"
        assert chunk.id == f"paper_doc_chunk_{index}"
        assert chunk.end_token - chunk.start_token <= 20
    for left, right in zip(chunks, chunks[1:]):
        assert right.start_token == left.end_token - 4


def test_multilingual_token_chunks_are_exact_source_substrings():
    text = ("中文论文GraphRAG：实体关系。🙂 Alpha-Beta. " * 20)
    chunks = TextChunker(
        chunk_size=17, chunk_overlap=3, unit="tokens"
    ).chunk_text(text, "cn")
    assert chunks
    assert all(chunk.text == text[chunk.start_char:chunk.end_char] for chunk in chunks)
    assert all("\ufffd" not in chunk.text for chunk in chunks)


def test_legacy_character_mode_has_source_spans():
    text = "One sentence. Two sentences. Three sentences."
    chunks = TextChunker(
        chunk_size=23, chunk_overlap=4, unit="characters"
    ).chunk_text(text, "legacy")
    assert len(chunks) >= 2
    assert all(chunk.text == text[chunk.start_char:chunk.end_char] for chunk in chunks)
    assert all(chunk.start_token is None for chunk in chunks)


def test_chunk_schema_rejects_broken_provenance():
    with pytest.raises(ValueError, match="character span must match"):
        TextChunk(
            id="chunk", text="actual", doc_id="doc", chunk_index=0,
            start_char=0, end_char=99,
        )
    with pytest.raises(ValueError, match="needs both"):
        TextChunk(id="chunk", text="actual", doc_id="doc", chunk_index=0,
                  start_token=0)
