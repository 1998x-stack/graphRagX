import pytest
from pydantic import ValidationError

from models.schemas import IndexRequest, QueryRequest


def test_index_id_is_safe_for_filesystem_storage():
    with pytest.raises(ValidationError):
        IndexRequest(index_id="../../etc/passwd", documents=["hello"])


def test_query_modes_are_explicit():
    request = QueryRequest(index_id="demo", query="hello", mode="basic", top_k=3)
    assert request.mode == "basic"

    drift = QueryRequest(index_id="demo", query="hello", mode="drift", top_k=3)
    assert drift.mode == "drift"

    with pytest.raises(ValidationError):
        QueryRequest(index_id="demo", query="hello", mode="unknown")
