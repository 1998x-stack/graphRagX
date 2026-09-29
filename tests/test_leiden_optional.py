import pytest

pytest.importorskip("igraph")

from core.community import CommunityDetector
from models.graph import KnowledgeGraph
from models.schemas import Entity, Relation


def test_optional_leiden_backend_covers_all_entities():
    kg = KnowledgeGraph()
    for name in ("A", "B", "C", "D"):
        kg.add_entity(Entity(name=name, type="CONCEPT", description=name))
    kg.add_relation(
        Relation(
            source="A",
            target="B",
            relation_type="RELATED",
            description="cluster one",
        )
    )
    kg.add_relation(
        Relation(
            source="C",
            target="D",
            relation_type="RELATED",
            description="cluster two",
        )
    )

    detector = CommunityDetector(max_levels=1)
    communities = detector.detect_communities(kg, algorithm="leiden")

    covered = {
        entity
        for community in communities
        for entity in community.entities
    }
    assert covered == {"A", "B", "C", "D"}
    assert all(community.level == 0 for community in communities)
