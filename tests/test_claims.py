import pytest

import core.claims as claims_module
from config import settings
from core.claims import ClaimExtractor
from models.graph import KnowledgeGraph
from models.schemas import Entity, TextChunk


@pytest.mark.asyncio
async def test_claim_extraction_canonicalizes_subject_and_provenance(monkeypatch):
    kg = KnowledgeGraph()
    kg.add_entity(
        Entity(
            name="Acme",
            type="ORGANIZATION",
            description="Company",
            aliases=["ACME"],
        )
    )
    kg.add_entity(Entity(name="Bob", type="PERSON", description="Employee"))

    async def fake_generate(prompt, task="general", **kwargs):
        del prompt, kwargs
        assert task == "claim_extraction"
        return (
            '{"claims":[{'
            '"type":"EMPLOYMENT",'
            '"description":"Acme employs Bob",'
            '"subject":"ACME",'
            '"object":"Bob",'
            '"status":"TRUE",'
            '"start_date":null,'
            '"end_date":null,'
            '"source_text":"Acme employs Bob"'
            '}]}'
        )

    monkeypatch.setattr(settings, "CLAIM_EXTRACTION_ENABLED", True)
    monkeypatch.setattr(claims_module.llm_service, "generate", fake_generate)

    claims = await ClaimExtractor().extract(
        [
            TextChunk(
                id="chunk_0",
                text="Acme employs Bob",
                doc_id="doc",
                chunk_index=0,
            )
        ],
        kg,
    )

    assert len(claims) == 1
    claim = claims[0]
    assert claim.subject_id == "Acme"
    assert claim.object_id == "Bob"
    assert claim.text_unit_id == "chunk_0"
    assert claim.status == "TRUE"
