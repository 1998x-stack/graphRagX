"""Paper 2404.16130 regression tests: complete cuts and corpus-wide map input."""
import pytest

from config import settings
from core.community import CommunityDetector
from models.schemas import Community, GraphData
from utils.token_budget import token_budget
from workflows.query_workflow import QueryWorkflow


def _hierarchy():
    return [
        Community(id="A", level=0, entities=["a", "b"], size=2),
        Community(id="B", level=0, entities=["c"], size=1),
        Community(id="A1", level=1, parent_id="A", entities=["a"], size=1),
        Community(id="A2", level=1, parent_id="A", entities=["b"], size=1),
    ]


def test_projected_hierarchy_cut_covers_each_entity_once():
    hierarchy = _hierarchy()
    assert [item.id for item in CommunityDetector.project_level(hierarchy, 0)] == [
        "A", "B"
    ]
    cut = CommunityDetector.project_level(hierarchy, 1)
    assert {item.id for item in cut} == {"B", "A1", "A2"}
    nodes = [node for community in cut for node in community.entities]
    assert sorted(nodes) == ["a", "b", "c"]
    assert len(nodes) == len(set(nodes))
    assert {item.id for item in CommunityDetector.project_level(hierarchy, 99)} == {
        "B", "A1", "A2"
    }


def test_partition_rejects_overlapping_or_missing_nodes(monkeypatch):
    import networkx as nx

    detector = CommunityDetector()
    graph = nx.Graph()
    graph.add_nodes_from(["a", "b", "c"])
    monkeypatch.setattr(
        detector, "_partition_louvain",
        lambda graph, resolution: [{"a", "b"}, {"b", "c"}],
    )
    with pytest.raises(ValueError, match="overlapping"):
        detector._partition_graph(graph, 1.0, "louvain")

    monkeypatch.setattr(
        detector, "_partition_louvain",
        lambda graph, resolution: [{"a"}, {"b"}],
    )
    with pytest.raises(ValueError, match="does not cover"):
        detector._partition_graph(graph, 1.0, "louvain")


def test_coverage_budget_fairly_preserves_all_reports():
    segments = [
        ("a", "alpha " * 40),
        ("b", "beta " * 40),
        ("c", "gamma " * 40),
    ]
    budgeted = token_budget.coverage_select(
        segments, max_tokens=90, min_tokens_per_segment=5
    )
    assert [item[0] for item in budgeted] == ["a", "b", "c"]
    assert all(item[1] for item in budgeted)
    assert sum(token_budget.count(text) for _, text in budgeted) <= 90
    with pytest.raises(ValueError, match="coverage budget"):
        token_budget.coverage_select(
            segments, max_tokens=5, min_tokens_per_segment=10
        )


@pytest.mark.asyncio
async def test_global_covers_entire_cut_even_when_top_k_is_one():
    hierarchy = _hierarchy()
    for community in hierarchy:
        community.summary = f"Verified evidence for {community.id}. " * 10
    data = GraphData(communities=hierarchy)
    workflow = QueryWorkflow()
    first = await workflow._retrieve_community_reports(
        "What are the themes?",
        data,
        top_k=1,
        community_level=1,
        max_tokens=240,
        selection_mode="coverage",
    )
    second = await workflow._retrieve_community_reports(
        "What are the themes?",
        data,
        top_k=1,
        community_level=1,
        max_tokens=240,
        selection_mode="coverage",
    )
    assert first == second
    assert {item["id"] for item in first["community_summaries"]} == {
        "B", "A1", "A2"
    }
    assert first["community_report_coverage"] == 1.0
    assert first["community_cut_size"] == 3
    assert sum(
        token_budget.count(item["summary"]) for item in first["community_summaries"]
    ) <= 240


@pytest.mark.asyncio
async def test_global_fails_closed_when_coverage_is_unaffordable(monkeypatch):
    monkeypatch.setattr(settings, "GLOBAL_MAX_REPORTS", 2)
    data = GraphData(communities=[
        Community(id=f"C{index}", level=0, entities=[str(index)], summary="report")
        for index in range(3)
    ])
    with pytest.raises(ValueError, match="GLOBAL_MAX_REPORTS"):
        await QueryWorkflow()._retrieve_community_reports(
            "themes", data, top_k=1, community_level=0,
            max_tokens=100, selection_mode="coverage"
        )


@pytest.mark.asyncio
async def test_coverage_rejects_missing_community_summaries():
    data = GraphData(communities=[
        Community(id="C", level=0, entities=["a"], summary=None)
    ])
    with pytest.raises(ValueError, match="missing community reports"):
        await QueryWorkflow()._retrieve_community_reports(
            "themes", data, top_k=5, community_level=0,
            max_tokens=100, selection_mode="coverage"
        )
