from pathlib import Path

import pytest

from models.schemas import Entity, GraphData
from services.storage_service import StorageService


def test_storage_rejects_path_traversal():
    with pytest.raises(ValueError):
        StorageService.validate_index_id("../escape")


def test_storage_round_trip(tmp_path: Path):
    storage = StorageService()
    storage.graphs_dir = tmp_path / "graphs"
    storage.graphs_dir.mkdir(parents=True)
    data = GraphData(entities={"A": Entity(name="A", type="CONCEPT", description="x")})
    storage.save_graph("demo", data)
    loaded = storage.load_graph("demo")
    assert loaded.entities["A"].description == "x"
    assert storage.list_indexes() == ["demo"]
