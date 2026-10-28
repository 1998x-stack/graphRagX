"""Community summaries include claim evidence and graph prominence ordering."""
import pytest

from core.summarization import CommunitySummarizer
from models.graph import KnowledgeGraph
from models.schemas import Claim, Community, Entity, Relation


@pytest.mark.asyncio
async def test_community_report_contains_claim_provenance_and_prominence(monkeypatch):
    kg = KnowledgeGraph()
    for name in ("A", "B", "C"):
        kg.add_entity(Entity(name=name, type="CONCEPT", description=f"entity {name}"))
    kg.add_relation(Relation(
        source="A", target="B", relation_type="SUPPORTS", description="A supports B",
        source_chunk_ids=["doc_0_chunk_0"],
    ))
    kg.add_relation(Relation(
        source="A", target="C", relation_type="RELATES", description="A relates C",
        source_chunk_ids=["doc_1_chunk_0"],
    ))
    claim = Claim(
        id="claim_abc",
        subject_id="A",
        object_id="B",
        status="TRUE",
        description="A and B are connected",
        text_unit_id="doc_0_chunk_0",
    )
    observed = []

    async def fake_generate(prompt, **kwargs):
        observed.append(prompt)
        return "Verified summary"

    monkeypatch.setattr("core.summarization.llm_service.generate", fake_generate)
    community = Community(id="L0_C0", level=0, entities=["B", "C", "A"])
    summary = await CommunitySummarizer().summarize_community(
        community, kg, claims=[claim]
    )
    assert summary == "Verified summary"
    prompt = observed[0]
    assert "claim_abc" in prompt
    assert "doc_0_chunk_0" in prompt
    assert prompt.index("A (CONCEPT)") < prompt.index("B (CONCEPT)")


@pytest.mark.asyncio
async def test_bottom_up_claims_do_not_break_child_report_order(monkeypatch):
    kg = KnowledgeGraph()
    kg.add_entity(Entity(name="A", type="CONCEPT", description="A"))
    claim = Claim(
        id="claim_1", subject_id="A", description="fact", text_unit_id="chunk_1"
    )
    levels = [
        Community(id="parent", level=0, entities=["A"]),
        Community(id="child", level=1, parent_id="parent", entities=["A"]),
    ]
    calls = []

    async def fake_generate(prompt, **kwargs):
        calls.append(prompt)
        return f"summary {len(calls)}"

    monkeypatch.setattr("core.summarization.llm_service.generate", fake_generate)
    result = await CommunitySummarizer().summarize_communities(
        levels, kg, claims=[claim]
    )
    assert result[0].summary == "summary 2"
    assert result[1].summary == "summary 1"
    assert all("claim_1" in prompt for prompt in calls)
