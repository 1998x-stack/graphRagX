from core.entity_resolution import EntityResolver, normalize_entity_surface
from models.schemas import Entity, ExtractionResult, Relation


def test_normalized_exact_resolution_remaps_aliases_and_relations():
    resolver = EntityResolver()
    results = [
        ExtractionResult(
            chunk_id="c1",
            entities=[
                Entity(
                    name="Acme",
                    type="ORGANIZATION",
                    description="Company",
                    source_chunk_ids=["c1"],
                )
            ],
        ),
        ExtractionResult(
            chunk_id="c2",
            entities=[
                Entity(
                    name="ACME!",
                    type="ORGANIZATION",
                    description="Technology company",
                    source_chunk_ids=["c2"],
                ),
                Entity(
                    name="Bob",
                    type="PERSON",
                    description="Employee",
                    source_chunk_ids=["c2"],
                ),
            ],
            relations=[
                Relation(
                    source="ACME!",
                    target="Bob",
                    relation_type="EMPLOYS",
                    description="Acme employs Bob",
                    source_chunk_ids=["c2"],
                )
            ],
        ),
    ]

    resolved, report = resolver.resolve(results)

    assert resolved[0].entities[0].name == "Acme"
    assert resolved[1].entities[0].name == "Acme"
    assert resolved[1].relations[0].source == "Acme"
    assert report["aliases_merged"] == 1
    assert normalize_entity_surface("  ACME! ") == "acme"


def test_resolution_does_not_cross_entity_types():
    resolver = EntityResolver()
    results = [
        ExtractionResult(
            chunk_id="c1",
            entities=[
                Entity(name="Mercury", type="PLANET", description="planet"),
                Entity(name="MERCURY", type="ORGANIZATION", description="org"),
            ],
        )
    ]
    resolved, report = resolver.resolve(results)
    assert len({(e.name, e.type) for e in resolved[0].entities}) == 2
    assert report["canonical_entities"] == 2
