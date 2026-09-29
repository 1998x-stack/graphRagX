from core.community import CommunityDetector
from models.graph import KnowledgeGraph
from models.schemas import Community, Entity


def test_hierarchy_builds_parent_child_relationships(monkeypatch):
    kg = KnowledgeGraph()
    for name in ("A", "B", "C", "D"):
        kg.add_entity(Entity(name=name, type="CONCEPT", description=name))

    detector = CommunityDetector(
        max_levels=2,
        max_cluster_size=1,
        resolution_multiplier=2.0,
    )

    def scripted_partition(graph, resolution, algorithm):
        del resolution, algorithm
        nodes = set(graph.nodes())
        if nodes == {"A", "B", "C", "D"}:
            return [{"A", "B"}, {"C", "D"}]
        if nodes == {"A", "B"}:
            return [{"A"}, {"B"}]
        if nodes == {"C", "D"}:
            return [{"C"}, {"D"}]
        return [nodes]

    monkeypatch.setattr(detector, "_partition_graph", scripted_partition)
    communities = detector.detect_communities(kg)

    level_zero = [item for item in communities if item.level == 0]
    level_one = [item for item in communities if item.level == 1]
    assert len(level_zero) == 2
    assert len(level_one) == 4

    by_id = {item.id: item for item in communities}
    for child in level_one:
        assert child.parent_id is not None
        assert set(child.entities).issubset(
            set(by_id[child.parent_id].entities)
        )


def test_choose_level_falls_back_to_nearest_available_lower_level():
    communities = [
        Community(id="a", level=0),
        Community(id="b", level=2),
    ]
    assert CommunityDetector.choose_level(communities, 1) == 0
    assert CommunityDetector.choose_level(communities, 3) == 2
