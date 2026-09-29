"""Unified query workflow for local, global, and basic retrieval."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph

from config import settings
from core.community import community_detector
from models.graph import KnowledgeGraph
from models.schemas import GraphData, TextChunk
from prompts.summary_prompts import (
    create_basic_search_prompt,
    create_global_map_prompt,
    create_global_reduce_prompt,
    create_local_search_prompt,
)
from services.embedding_service import embedding_service
from services.llm_service import llm_service
from services.storage_service import storage_service
from utils.concurrency import concurrency_controller
from utils.json_extractor import extract_json
from utils.logger import log, log_exception
from utils.token_budget import token_budget


QueryMode = Literal["local", "global", "basic"]


class QueryWorkflowState(TypedDict, total=False):
    query: str
    index_id: str
    mode: QueryMode
    top_k: int
    community_level: int | None
    graph_data: GraphData
    kg_object: KnowledgeGraph
    relevant_context: Dict[str, Any]
    answer: str
    current_step: str
    error: str | None


class QueryWorkflow:
    def __init__(self):
        self.graph = self._build_graph()

    @staticmethod
    def _failure(exc: Exception, context: str) -> dict:
        log_exception(exc, context)
        return {"current_step": "error", "error": str(exc)}

    @staticmethod
    def _route_mode(state: QueryWorkflowState) -> str:
        if state.get("error"):
            return "end"
        return state["mode"]

    @staticmethod
    def _route_after_retrieval(state: QueryWorkflowState) -> str:
        return "end" if state.get("error") else "continue"

    def _build_graph(self):
        workflow = StateGraph(QueryWorkflowState)
        workflow.add_node("load_graph", self.load_graph)
        workflow.add_node("local_search", self.local_search)
        workflow.add_node("global_search", self.global_search)
        workflow.add_node("basic_search", self.basic_search)
        workflow.add_node("generate_local_answer", self.generate_local_answer)
        workflow.add_node("generate_global_answer", self.generate_global_answer)
        workflow.add_node("generate_basic_answer", self.generate_basic_answer)

        workflow.add_edge(START, "load_graph")
        workflow.add_conditional_edges(
            "load_graph",
            self._route_mode,
            {
                "local": "local_search",
                "global": "global_search",
                "basic": "basic_search",
                "end": END,
            },
        )
        for retrieval, generator in (
            ("local_search", "generate_local_answer"),
            ("global_search", "generate_global_answer"),
            ("basic_search", "generate_basic_answer"),
        ):
            workflow.add_conditional_edges(
                retrieval,
                self._route_after_retrieval,
                {"continue": generator, "end": END},
            )
            workflow.add_edge(generator, END)
        return workflow.compile()

    async def load_graph(self, state: QueryWorkflowState) -> dict:
        try:
            graph_data = storage_service.load_graph(state["index_id"])
            return {
                "graph_data": graph_data,
                "kg_object": KnowledgeGraph.from_graph_data(graph_data),
                "current_step": "graph_loaded",
                "error": None,
            }
        except Exception as exc:
            return self._failure(exc, "load_graph")

    @staticmethod
    def _stored_embeddings_compatible(graph_data: GraphData) -> bool:
        metadata = graph_data.metadata
        return (
            metadata.get("embedding_provider") == settings.EMBEDDING_PROVIDER
            and metadata.get("embedding_model") == embedding_service.model
            and metadata.get("embedding_dimension") == embedding_service.dimension
        )

    async def _rank_named_vectors(
        self,
        query: str,
        names: List[str],
        vectors_by_name: Dict[str, List[float]],
        fallback_texts: List[str],
        top_k: int,
        use_stored: bool = True,
    ) -> List[tuple[str, float]]:
        if not names:
            return []

        query_vector = await embedding_service.embed_text(query)
        stored_usable = use_stored and all(
            name in vectors_by_name
            and len(vectors_by_name[name]) == len(query_vector)
            for name in names
        )
        if stored_usable:
            vectors = [
                np.asarray(vectors_by_name[name], dtype=np.float32)
                for name in names
            ]
        else:
            vectors = await embedding_service.embed_texts(fallback_texts)

        scored = [
            (
                name,
                embedding_service.cosine_similarity(query_vector, vector),
            )
            for name, vector in zip(names, vectors)
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[: min(top_k, len(scored))]

    async def _rank_chunks(
        self,
        query: str,
        chunks: List[TextChunk],
        graph_data: GraphData,
        top_k: int,
    ) -> List[TextChunk]:
        if not chunks:
            return []

        query_vector = await embedding_service.embed_text(query)
        stored_usable = self._stored_embeddings_compatible(graph_data) and all(
            chunk.id in graph_data.chunk_embeddings
            and len(graph_data.chunk_embeddings[chunk.id]) == len(query_vector)
            for chunk in chunks
        )
        if stored_usable:
            vectors = [
                np.asarray(graph_data.chunk_embeddings[chunk.id], dtype=np.float32)
                for chunk in chunks
            ]
        else:
            vectors = await embedding_service.embed_texts(
                [chunk.text for chunk in chunks]
            )

        scored = [
            (
                chunk,
                embedding_service.cosine_similarity(query_vector, vector),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        scored.sort(key=lambda item: (-item[1], item[0].id))
        return [chunk for chunk, _ in scored[: min(top_k, len(scored))]]

    async def local_search(self, state: QueryWorkflowState) -> dict:
        try:
            kg = state["kg_object"]
            graph_data = state["graph_data"]
            top_k = min(state.get("top_k", 5), settings.MAX_QUERY_TOP_K)
            entity_names = sorted(kg.entities)
            ranked_entities = await self._rank_named_vectors(
                state["query"],
                entity_names,
                graph_data.entity_embeddings,
                [f"{name}: {kg.entities[name].description}" for name in entity_names],
                top_k,
                use_stored=self._stored_embeddings_compatible(graph_data),
            )
            selected = [name for name, _ in ranked_entities]

            relevant_names = set(selected)
            for name in selected:
                relevant_names.update(
                    kg.get_neighbors(
                        name,
                        depth=settings.RETRIEVAL_NEIGHBOR_DEPTH,
                    )
                )

            entities = [
                kg.entities[name].model_dump()
                for name in sorted(relevant_names)
                if name in kg.entities
            ]
            relations = [
                relation.model_dump()
                for relation in kg.relations
                if relation.source in relevant_names
                and relation.target in relevant_names
            ]

            source_chunk_ids = {
                chunk_id
                for name in relevant_names
                if name in kg.entities
                for chunk_id in kg.entities[name].source_chunk_ids
            }
            chunks_by_id = {
                chunk.id: chunk
                for chunk in graph_data.text_chunks
            }
            source_chunks = [
                chunks_by_id[chunk_id]
                for chunk_id in source_chunk_ids
                if chunk_id in chunks_by_id
            ]
            ranked_chunks = await self._rank_chunks(
                state["query"],
                source_chunks,
                graph_data,
                top_k=settings.MAX_CONTEXT_CHUNKS,
            )
            packed_segments = token_budget.select(
                [(chunk.id, chunk.text) for chunk in ranked_chunks],
                settings.LOCAL_MAX_DATA_TOKENS,
            )
            packed_text = dict(packed_segments)
            text_units = [
                chunk.model_copy(
                    update={"text": packed_text[chunk.id]}
                ).model_dump()
                for chunk in ranked_chunks
                if chunk.id in packed_text
            ]

            context = {
                "entities": entities,
                "relations": relations,
                "text_units": text_units,
                "seed_entities": [
                    {"name": name, "relevance": score}
                    for name, score in ranked_entities
                ],
            }
            return {
                "relevant_context": context,
                "current_step": "context_retrieved",
            }
        except Exception as exc:
            return self._failure(exc, "local_search")

    async def global_search(self, state: QueryWorkflowState) -> dict:
        try:
            graph_data = state["graph_data"]
            communities = [
                community
                for community in graph_data.communities
                if community.summary
            ]
            requested_level = state.get("community_level")
            if requested_level is None:
                requested_level = settings.GLOBAL_COMMUNITY_LEVEL
            chosen_level = community_detector.choose_level(
                communities,
                requested_level,
            )
            level_communities = [
                community
                for community in communities
                if community.level == chosen_level
            ]
            ids = [community.id for community in level_communities]
            limit = min(
                state.get("top_k", 5),
                settings.GLOBAL_TOP_K,
                len(ids),
            )
            ranked = await self._rank_named_vectors(
                state["query"],
                ids,
                graph_data.community_embeddings,
                [community.summary or "" for community in level_communities],
                limit,
                use_stored=self._stored_embeddings_compatible(graph_data),
            )
            by_id = {
                community.id: community
                for community in level_communities
            }
            budgeted = token_budget.select(
                [
                    (community_id, by_id[community_id].summary or "")
                    for community_id, _ in ranked
                ],
                settings.GLOBAL_MAX_DATA_TOKENS,
            )
            budgeted_text = dict(budgeted)
            reports = [
                {
                    "id": community_id,
                    "summary": budgeted_text[community_id],
                    "size": by_id[community_id].size,
                    "level": by_id[community_id].level,
                    "parent_id": by_id[community_id].parent_id,
                    "relevance": score,
                }
                for community_id, score in ranked
                if community_id in budgeted_text
            ]
            return {
                "relevant_context": {
                    "community_summaries": reports,
                    "community_level": chosen_level,
                },
                "current_step": "context_retrieved",
            }
        except Exception as exc:
            return self._failure(exc, "global_search")

    @staticmethod
    def _parse_global_map_points(
        raw_response: str,
        source_reports: List[str],
    ) -> List[Dict[str, Any]]:
        data = extract_json(raw_response)
        points: List[Dict[str, Any]] = []
        if isinstance(data, dict) and isinstance(data.get("points"), list):
            for point in data["points"]:
                if not isinstance(point, dict):
                    continue
                description = str(point.get("description", "")).strip()
                if not description:
                    continue
                try:
                    score = float(point.get("score", 0))
                except (TypeError, ValueError):
                    score = 0.0
                points.append(
                    {
                        "description": description,
                        "score": max(0.0, min(100.0, score)),
                        "source_reports": source_reports,
                    }
                )
        if not points and raw_response.strip():
            points.append(
                {
                    "description": raw_response.strip(),
                    "score": 1.0,
                    "source_reports": source_reports,
                }
            )
        return points

    async def basic_search(self, state: QueryWorkflowState) -> dict:
        try:
            graph_data = state["graph_data"]
            chunks = graph_data.text_chunks
            if not chunks:
                return {
                    "relevant_context": {"text_units": []},
                    "current_step": "context_retrieved",
                }
            ranked = await self._rank_chunks(
                state["query"],
                chunks,
                graph_data,
                top_k=min(
                    state.get("top_k", 5),
                    settings.MAX_QUERY_TOP_K,
                ),
            )
            return {
                "relevant_context": {
                    "text_units": [
                        chunk.model_dump()
                        for chunk in ranked
                    ]
                },
                "current_step": "context_retrieved",
            }
        except Exception as exc:
            return self._failure(exc, "basic_search")

    async def generate_local_answer(self, state: QueryWorkflowState) -> dict:
        try:
            context = state["relevant_context"]
            prompt = create_local_search_prompt(
                state["query"],
                context.get("entities", []),
                context.get("relations", []),
                context.get("text_units", []),
            )
            answer = await llm_service.generate(
                prompt,
                task="local_query_answer",
            )
            return {
                "answer": answer.strip(),
                "current_step": "completed",
            }
        except Exception as exc:
            return self._failure(exc, "generate_local_answer")

    async def generate_global_answer(self, state: QueryWorkflowState) -> dict:
        try:
            context = dict(state["relevant_context"])
            reports = context.get("community_summaries", [])
            if not reports:
                return {
                    "answer": (
                        "No community reports were available at the selected "
                        "hierarchy level."
                    ),
                    "current_step": "completed",
                }

            batches = token_budget.batches(
                [
                    (report["id"], report.get("summary", ""))
                    for report in reports
                ],
                settings.GLOBAL_MAP_BATCH_TOKENS,
            )

            async def map_one(item: tuple[int, list[tuple[str, str]]]):
                batch_index, batch = item
                batch_reports = [
                    {"id": report_id, "summary": summary}
                    for report_id, summary in batch
                ]
                prompt = create_global_map_prompt(
                    state["query"],
                    batch_reports,
                )
                raw = await llm_service.generate(
                    prompt,
                    task=f"global_map_{batch_index}",
                )
                return self._parse_global_map_points(
                    raw,
                    [report_id for report_id, _ in batch],
                )

            mapped = await concurrency_controller.map_async(
                func=map_one,
                items=list(enumerate(batches)),
                return_exceptions=False,
            )
            points = [
                point
                for batch_points in mapped
                for point in batch_points
                if point.get("score", 0) > 0
            ]
            points.sort(
                key=lambda point: (
                    -float(point.get("score", 0)),
                    point.get("description", ""),
                )
            )
            if not points:
                return {
                    "answer": (
                        "The selected community reports did not contain enough "
                        "relevant evidence to answer the question."
                    ),
                    "current_step": "completed",
                }

            point_lookup = {
                f"point_{index}": point
                for index, point in enumerate(points)
            }
            selected_point_text = token_budget.select(
                [
                    (point_id, point["description"])
                    for point_id, point in point_lookup.items()
                ],
                settings.GLOBAL_REDUCE_DATA_TOKENS,
            )
            reduce_points = [
                {
                    **point_lookup[point_id],
                    "description": description,
                }
                for point_id, description in selected_point_text
            ]
            prompt = create_global_reduce_prompt(
                state["query"],
                reduce_points,
            )
            answer = await llm_service.generate(
                prompt,
                task="global_reduce",
            )
            context["map_points"] = reduce_points
            return {
                "answer": answer.strip(),
                "relevant_context": context,
                "current_step": "completed",
            }
        except Exception as exc:
            return self._failure(exc, "generate_global_answer")

    async def generate_basic_answer(self, state: QueryWorkflowState) -> dict:
        try:
            prompt = create_basic_search_prompt(
                state["query"],
                state["relevant_context"].get("text_units", []),
            )
            answer = await llm_service.generate(
                prompt,
                task="basic_query_answer",
            )
            return {
                "answer": answer.strip(),
                "current_step": "completed",
            }
        except Exception as exc:
            return self._failure(exc, "generate_basic_answer")

    async def run(
        self,
        query: str,
        index_id: str,
        mode: QueryMode = "local",
        top_k: int = 5,
        community_level: int | None = None,
    ) -> dict:
        initial: QueryWorkflowState = {
            "query": query,
            "index_id": index_id,
            "mode": mode,
            "top_k": top_k,
            "community_level": community_level,
            "relevant_context": {},
            "answer": "",
            "current_step": "init",
            "error": None,
        }
        final_state = await self.graph.ainvoke(initial)
        log.info(
            "Query workflow {}:{} finished at {}",
            index_id,
            mode,
            final_state.get("current_step"),
        )
        return final_state


query_workflow = QueryWorkflow()
