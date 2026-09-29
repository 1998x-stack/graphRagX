"""Summarize repeated entity and relation descriptions without losing mentions."""
from __future__ import annotations

from config import settings
from models.graph import KnowledgeGraph
from prompts.quality_prompts import (
    create_entity_description_summary_prompt,
    create_relation_description_summary_prompt,
)
from services.llm_service import llm_service
from utils.concurrency import concurrency_controller
from utils.logger import log, log_exception


class DescriptionSummarizer:
    @staticmethod
    def _mentions(description: str, mentions: list[str]) -> list[str]:
        values = list(dict.fromkeys(
            item.strip()
            for item in ([description] + mentions)
            if item and item.strip()
        ))
        return values[: settings.DESCRIPTION_SUMMARY_MAX_MENTIONS]

    async def summarize_graph(self, kg: KnowledgeGraph) -> KnowledgeGraph:
        if not settings.DESCRIPTION_SUMMARIZATION_ENABLED:
            return kg

        entities = sorted(kg.entities.values(), key=lambda item: item.name)

        async def summarize_entity(entity):
            mentions = self._mentions(
                entity.description,
                entity.description_mentions,
            )
            entity.description_mentions = mentions
            if len(mentions) <= 1 or settings.LLM_PROVIDER == "mock":
                entity.description = " | ".join(mentions)
                return entity

            try:
                prompt = create_entity_description_summary_prompt(
                    entity.name,
                    entity.type,
                    mentions,
                )
                summary = await llm_service.generate(
                    prompt,
                    task=f"entity_description_summary_{entity.name}",
                )
                entity.description = summary.strip() or " | ".join(mentions)
            except Exception as exc:
                log_exception(exc, f"summarize_entity({entity.name})")
                entity.description = " | ".join(mentions)
            return entity

        summarized_entities = await concurrency_controller.map_async(
            func=summarize_entity,
            items=entities,
            return_exceptions=False,
        )

        relations = sorted(
            kg.relations,
            key=lambda item: (item.source, item.target, item.relation_type),
        )

        async def summarize_relation(relation):
            mentions = self._mentions(
                relation.description,
                relation.description_mentions,
            )
            relation.description_mentions = mentions
            if len(mentions) <= 1 or settings.LLM_PROVIDER == "mock":
                relation.description = " | ".join(mentions)
                return relation

            try:
                prompt = create_relation_description_summary_prompt(
                    relation.source,
                    relation.relation_type,
                    relation.target,
                    mentions,
                )
                summary = await llm_service.generate(
                    prompt,
                    task=(
                        "relation_description_summary_"
                        f"{relation.source}_{relation.relation_type}_{relation.target}"
                    ),
                )
                relation.description = summary.strip() or " | ".join(mentions)
            except Exception as exc:
                log_exception(
                    exc,
                    f"summarize_relation({relation.source},{relation.target})",
                )
                relation.description = " | ".join(mentions)
            return relation

        summarized_relations = await concurrency_controller.map_async(
            func=summarize_relation,
            items=relations,
            return_exceptions=False,
        )

        rebuilt = KnowledgeGraph()
        for entity in summarized_entities:
            rebuilt.add_entity(entity)
        for relation in summarized_relations:
            rebuilt.add_relation(relation)
        log.info(
            "Description summarization completed: entities={} relations={}",
            len(rebuilt.entities),
            len(rebuilt.relations),
        )
        return rebuilt


description_summarizer = DescriptionSummarizer()
