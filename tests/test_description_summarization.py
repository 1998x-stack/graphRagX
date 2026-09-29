import pytest

from core.description_summarization import DescriptionSummarizer
from models.graph import KnowledgeGraph
from models.schemas import Entity, Relation


@pytest.mark.asyncio
async def test_description_mentions_are_preserved_and_merged_offline():
    kg = KnowledgeGraph()
    kg.add_entity(
        Entity(
            name="Acme",
            type="ORGANIZATION",
            description="Builds retrieval systems",
            source_chunk_ids=["c1"],
        )
    )
    kg.add_entity(
        Entity(
            name="Acme",
            type="ORGANIZATION",
            description="Develops graph software",
            source_chunk_ids=["c2"],
        )
    )
    kg.add_entity(
        Entity(
            name="Bob",
            type="PERSON",
            description="Engineer",
            source_chunk_ids=["c1"],
        )
    )
    kg.add_relation(
        Relation(
            source="Bob",
            target="Acme",
            relation_type="WORKS_FOR",
            description="Bob works at Acme",
            source_chunk_ids=["c1"],
        )
    )
    kg.add_relation(
        Relation(
            source="Bob",
            target="Acme",
            relation_type="WORKS_FOR",
            description="Bob is employed by Acme",
            source_chunk_ids=["c2"],
        )
    )

    summarized = await DescriptionSummarizer().summarize_graph(kg)

    entity = summarized.entities["Acme"]
    relation = summarized.relations[0]
    assert entity.mention_count == 2
    assert len(entity.description_mentions) == 2
    assert "Builds retrieval systems" in entity.description
    assert "Develops graph software" in entity.description
    assert relation.mention_count == 2
    assert len(relation.description_mentions) == 2
