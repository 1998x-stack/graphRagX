from models.graph import KnowledgeGraph
from models.schemas import Entity, Relation


def test_graph_preserves_direction_and_deduplicates_relations():
    kg = KnowledgeGraph()
    kg.add_entity(Entity(name="Alice", type="PERSON", description="Engineer", source_chunk_ids=["c1"]))
    kg.add_entity(Entity(name="Acme", type="ORG", description="Company", source_chunk_ids=["c1"]))
    kg.add_relation(Relation(source="Alice", target="Acme", relation_type="WORKS_FOR", description="works", source_chunk_ids=["c1"]))
    kg.add_relation(Relation(source="Alice", target="Acme", relation_type="WORKS_FOR", description="employed", source_chunk_ids=["c2"]))
    kg.add_relation(Relation(source="Acme", target="Alice", relation_type="EMPLOYS", description="employs", source_chunk_ids=["c1"]))

    assert len(kg.relations) == 2
    works_for = next(item for item in kg.relations if item.relation_type == "WORKS_FOR")
    assert works_for.weight == 2.0
    assert set(works_for.source_chunk_ids) == {"c1", "c2"}
    assert kg.graph.has_edge("Alice", "Acme", key="WORKS_FOR")
    assert kg.graph.has_edge("Acme", "Alice", key="EMPLOYS")
