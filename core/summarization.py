"""Bottom-up community report generation."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from models.graph import KnowledgeGraph
from models.schemas import Community
from prompts.summary_prompts import create_community_summary_prompt
from services.llm_service import llm_service
from utils.concurrency import concurrency_controller
from utils.logger import log, log_exception


class CommunitySummarizer:
    async def summarize_community(
        self,
        community: Community,
        kg: KnowledgeGraph,
        child_reports: list | None = None,
    ) -> str:
        try:
            entities = [
                entity.model_dump()
                for entity_name in community.entities
                if (entity := kg.get_entity(entity_name)) is not None
            ]
            entity_names = set(community.entities)
            relations = [
                relation.model_dump()
                for relation in kg.relations
                if relation.source in entity_names and relation.target in entity_names
            ]
            prompt = create_community_summary_prompt(
                community_id=community.id,
                entities=entities,
                relations=relations,
                level=community.level,
                child_reports=child_reports or [],
            )
            summary = await llm_service.generate(
                prompt=prompt,
                task=f"community_summary_{community.id}",
                save_response=True,
            )
            summary = summary.strip()
            if not summary:
                raise RuntimeError(f"Empty community report for {community.id}")
            return summary
        except Exception as exc:
            log_exception(exc, f"summarize_community({community.id})")
            raise

    async def summarize_communities(
        self,
        communities: List[Community],
        kg: KnowledgeGraph,
    ) -> List[Community]:
        """Generate reports from the deepest level upward."""
        if not communities:
            return []

        by_id: Dict[str, Community] = {
            community.id: community.model_copy(deep=True)
            for community in communities
        }
        children_by_parent: Dict[str, List[str]] = defaultdict(list)
        for community in by_id.values():
            if community.parent_id:
                children_by_parent[community.parent_id].append(community.id)

        levels = sorted({community.level for community in by_id.values()}, reverse=True)
        for level in levels:
            level_communities = sorted(
                (community for community in by_id.values() if community.level == level),
                key=lambda community: community.id,
            )

            async def summarize_single(community: Community) -> Community:
                children = [
                    by_id[child_id]
                    for child_id in sorted(children_by_parent.get(community.id, []))
                ]
                child_reports = [
                    {
                        "id": child.id,
                        "level": child.level,
                        "summary": child.summary,
                    }
                    for child in children
                    if child.summary
                ]
                updated = community.model_copy(deep=True)
                updated.summary = await self.summarize_community(
                    updated,
                    kg,
                    child_reports=child_reports,
                )
                return updated

            summarized = await concurrency_controller.map_async(
                func=summarize_single,
                items=level_communities,
                return_exceptions=False,
            )
            for community in summarized:
                by_id[community.id] = community

        result = sorted(by_id.values(), key=lambda community: (community.level, community.id))
        log.info(
            "Generated {} bottom-up community reports across {} level(s)",
            len(result),
            len(levels),
        )
        return result


community_summarizer = CommunitySummarizer()
