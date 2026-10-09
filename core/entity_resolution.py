"""Conservative deterministic entity resolution for extracted mentions."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Dict, List, Tuple

from config import settings
from models.schemas import Entity, ExtractionResult, Relation
from utils.logger import log


def normalize_entity_surface(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return " ".join(normalized.split())


class EntityResolver:
    """Resolve only high-confidence normalized-exact aliases.

    Matching is constrained by entity type. We intentionally avoid fuzzy matching
    because false merges are more damaging to graph retrieval than missed merges.
    """

    def resolve(
        self,
        extraction_results: List[ExtractionResult],
    ) -> Tuple[List[ExtractionResult], Dict[str, int | str]]:
        if settings.ENTITY_RESOLUTION_STRATEGY == "none":
            mentions = sum(len(result.entities) for result in extraction_results)
            return extraction_results, {
                "strategy": "none",
                "entity_mentions": mentions,
                "canonical_entities": mentions,
                "aliases_merged": 0,
            }

        counts: Counter[tuple[str, str, str]] = Counter()
        first_seen: Dict[tuple[str, str], str] = {}
        ordered_mentions: List[Entity] = []
        for result in extraction_results:
            for entity in result.entities:
                surface = normalize_entity_surface(entity.name)
                entity_type = entity.type.strip().upper()
                key = (entity_type, surface)
                first_seen.setdefault(key, entity.name.strip())
                counts[(entity_type, surface, entity.name.strip())] += entity.mention_count
                ordered_mentions.append(entity)

        canonical_by_key: Dict[tuple[str, str], str] = {}
        keys = {(entity.type.strip().upper(), normalize_entity_surface(entity.name)) for entity in ordered_mentions}
        for key in keys:
            candidates = [
                (count, name)
                for (entity_type, surface, name), count in counts.items()
                if (entity_type, surface) == key
            ]
            candidates.sort(key=lambda item: (-item[0], item[1] != first_seen[key], item[1].casefold()))
            canonical_by_key[key] = candidates[0][1]

        alias_to_canonical: Dict[tuple[str, str], str] = dict(canonical_by_key)
        raw_name_map: Dict[str, str] = {}
        for entity in ordered_mentions:
            key = (entity.type.strip().upper(), normalize_entity_surface(entity.name))
            raw_name_map[entity.name] = canonical_by_key[key]

        resolved: List[ExtractionResult] = []
        for result in extraction_results:
            entities = []
            for entity in result.entities:
                key = (entity.type.strip().upper(), normalize_entity_surface(entity.name))
                canonical = alias_to_canonical[key]
                description_mentions = entity.description_mentions or ([entity.description] if entity.description else [])
                entities.append(
                    entity.model_copy(
                        update={
                            "name": canonical,
                            "type": entity.type.strip().upper(),
                            "aliases": sorted(set(entity.aliases + [entity.name])),
                            "description_mentions": list(dict.fromkeys(description_mentions)),
                        }
                    )
                )

            relations = []
            for relation in result.relations:
                source = raw_name_map.get(relation.source, relation.source)
                target = raw_name_map.get(relation.target, relation.target)
                description_mentions = relation.description_mentions or (
                    [relation.description] if relation.description else []
                )
                relations.append(
                    relation.model_copy(
                        update={
                            "source": source,
                            "target": target,
                            "relation_type": relation.relation_type.strip().upper(),
                            "description_mentions": list(dict.fromkeys(description_mentions)),
                        }
                    )
                )

            resolved.append(
                result.model_copy(
                    update={
                        "entities": entities,
                        "relations": relations,
                    }
                )
            )

        canonical_count = len(set(canonical_by_key.values()))
        mention_count = len(ordered_mentions)
        report = {
            "strategy": settings.ENTITY_RESOLUTION_STRATEGY,
            "entity_mentions": mention_count,
            "canonical_entities": canonical_count,
            "aliases_merged": max(0, mention_count - canonical_count),
        }
        log.info("Entity resolution report: {}", report)
        return resolved, report


entity_resolver = EntityResolver()
