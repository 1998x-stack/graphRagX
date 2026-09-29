import pytest

from core.summarization import CommunitySummarizer
from models.graph import KnowledgeGraph
from models.schemas import Community


@pytest.mark.asyncio
async def test_bottom_up_summaries_feed_children_into_parent(monkeypatch):
    summarizer = CommunitySummarizer()
    communities = [
        Community(
            id="L0_C0",
            level=0,
            entities=["A", "B"],
            size=2,
        ),
        Community(
            id="L1_C0",
            level=1,
            parent_id="L0_C0",
            entities=["A"],
            size=1,
        ),
        Community(
            id="L1_C1",
            level=1,
            parent_id="L0_C0",
            entities=["B"],
            size=1,
        ),
    ]
    calls = []

    async def fake_summary(community, kg, child_reports=None):
        del kg
        calls.append(
            (
                community.id,
                [
                    (item["id"], item["summary"])
                    for item in (child_reports or [])
                ],
            )
        )
        return f"summary:{community.id}"

    monkeypatch.setattr(
        summarizer,
        "summarize_community",
        fake_summary,
    )

    result = await summarizer.summarize_communities(
        communities,
        KnowledgeGraph(),
    )
    by_id = {item.id: item for item in result}

    assert by_id["L1_C0"].summary == "summary:L1_C0"
    assert by_id["L1_C1"].summary == "summary:L1_C1"

    parent_call = next(
        child_reports
        for community_id, child_reports in calls
        if community_id == "L0_C0"
    )
    assert parent_call == [
        ("L1_C0", "summary:L1_C0"),
        ("L1_C1", "summary:L1_C1"),
    ]
