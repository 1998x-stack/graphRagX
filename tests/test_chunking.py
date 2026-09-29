import pytest

from core.chunking import TextChunker


def test_chunker_rejects_non_progressing_overlap():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=100)


def test_chunker_is_deterministic_and_progresses():
    chunker = TextChunker(chunk_size=30, chunk_overlap=5)
    text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."
    first = chunker.chunk_text(text, "doc")
    second = chunker.chunk_text(text, "doc")
    assert [item.model_dump() for item in first] == [item.model_dump() for item in second]
    assert 1 < len(first) < 10
    assert all(item.text for item in first)
