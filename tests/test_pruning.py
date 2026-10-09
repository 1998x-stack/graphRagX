from config import settings
from core.pruning import GraphPruner
from models.graph import KnowledgeGraph
from models.schemas import Entity, Relation


def build_graph():
    kg = KnowledgeGraph()
    for name in ("A", "B", "C"):
        kg.add_entity(Entity(name=name, type="CONCEPT", description=name))
    kg.add_relation(
        Relation(
            source="A",
            target="B",
            relation_type="RELATED",
            description="weak",
            weight=1.0,
        )
    )
    kg.add_relation(
        Relation(
            source="B",
            target="C",
            relation_type="RELATED",
            description="strong",
            weight=5.0,
        )
    )
    return kg


def test_edge_percentile_pruning_is_auditable(monkeypatch):
    monkeypatch.setattr(settings, "GRAPH_PRUNING_ENABLED", True)
    monkeypatch.setattr(settings, "PRUNE_MIN_EDGE_WEIGHT_PCT", 50.0)
    monkeypatch.setattr(settings, "PRUNE_MIN_NODE_FREQ", 1)
    monkeypatch.setattr(settings, "PRUNE_MIN_NODE_DEGREE", 0)
    monkeypatch.setattr(settings, "PRUNE_LCC_ONLY", False)
    monkeypatch.setattr(settings, "PRUNE_REMOVE_EGO_NODES", False)
    monkeypatch.setattr(settings, "PRUNE_MAX_NODE_FREQ_STD", None)
    monkeypatch.setattr(settings, "PRUNE_MAX_NODE_DEGREE_STD", None)

    pruned, report = GraphPruner().prune(build_graph())

    assert len(pruned.entities) == 3
    assert len(pruned.relations) == 1
    assert pruned.relations[0].description == "strong"
    assert report["removed_relations"] == 1
    assert report["edge_weight_threshold"] == 3.0


def test_lcc_only_keeps_deterministic_largest_component(monkeypatch):
    monkeypatch.setattr(settings, "GRAPH_PRUNING_ENABLED", True)
    monkeypatch.setattr(settings, "PRUNE_MIN_EDGE_WEIGHT_PCT", 0.0)
    monkeypatch.setattr(settings, "PRUNE_MIN_NODE_FREQ", 1)
    monkeypatch.setattr(settings, "PRUNE_MIN_NODE_DEGREE", 0)
    monkeypatch.setattr(settings, "PRUNE_LCC_ONLY", True)
    monkeypatch.setattr(settings, "PRUNE_REMOVE_EGO_NODES", False)
    monkeypatch.setattr(settings, "PRUNE_MAX_NODE_FREQ_STD", None)
    monkeypatch.setattr(settings, "PRUNE_MAX_NODE_DEGREE_STD", None)

    kg = build_graph()
    kg.add_entity(Entity(name="Z", type="CONCEPT", description="isolated"))
    pruned, _ = GraphPruner().prune(kg)
    assert set(pruned.entities) == {"A", "B", "C"}
