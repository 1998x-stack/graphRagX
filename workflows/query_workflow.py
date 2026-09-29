"""Unified LangGraph query workflow: local, global, and basic vector RAG."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph

from config import settings
from models.graph import KnowledgeGraph
from models.schemas import GraphData, TextChunk
from prompts.summary_prompts import (
    create_basic_search_prompt,
    create_global_search_prompt,
    create_local_search_prompt,
)
from services.embedding_service import embedding_service
from services.llm_service import llm_service
from services.storage_service import storage_service
from utils.logger import log, log_exception


QueryMode = Literal["local", "global", "basic"]


class QueryWorkflowState(TypedDict, total=False):
    query: str
    index_id: str
    mode: QueryMode
    top_k: int
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
    ) -> List[str]:
        if not names:
            return []
        query_vector = await embedding_service.embed_text(query)
        stored_usable = use_stored and all(
            name in vectors_by_name and len(vectors_by_name[name]) == len(query_vector)
            for name in names
        )
        if stored_usable:
            vectors = [np.asarray(vectors_by_name[name], dtype=np.float32) for name in names]
        else:
            vectors = await embedding_service.embed_texts(fallback_texts)
        indices = embedding_service.find_most_similar(query_vector, vectors, top_k=top_k)
        return [names[index] for index in indices]

    async def local_search(self, state: QueryWorkflowState) -> dict:
        try:
            kg = state["kg_object"]
            graph_data = state["graph_data"]
            top_k = min(state.get("top_k", 5), settings.MAX_QUERY_TOP_K)
            entity_names = sorted(kg.entities)
            selected = await self._rank_named_vectors(
                state["query"],
                entity_names,
                graph_data.entity_embeddings,
                [f"{name}: {kg.entities[name].description}" for name in entity_names],
                top_k,
                use_stored=self._stored_embeddings_compatible(graph_data),
            )

            relevant_names = set(selected)
            for name in selected:
                relevant_names.update(kg.get_neighbors(name, depth=settings.RETRIEVAL_NEIGHBOR_DEPTH))

            entities = [kg.entities[name].model_dump() for name in sorted(relevant_names) if name in kg.entities]
            relations = [
                relation.model_dump()
                for relation in kg.relations
                if relation.source in relevant_names and relation.target in relevant_names
            ]

            source_chunk_ids = {
                chunk_id
                for name in relevant_names
                if name in kg.entities
                for chunk_id in kg.entities[name].source_chunk_ids
            }
            chunks_by_id = {chunk.id: chunk for chunk in graph_data.text_chunks}
            source_chunks = [chunks_by_id[chunk_id] for chunk_id in sorted(source_chunk_ids) if chunk_id in chunks_by_id]
            source_chunks.sort(key=lambda chunk: (chunk.doc_id, chunk.chunk_index, chunk.id))
            source_chunks = source_chunks[: settings.MAX_CONTEXT_CHUNKS]

            context = {
                "entities": entities,
                "relations": relations,
                "text_units": [chunk.model_dump() for chunk in source_chunks],
                "seed_entities": selected,
            }
            return {"relevant_context": context, "current_step": "context_retrieved"}
        except Exception as exc:
            return self._failure(exc, "local_search")

    async def global_search(self, state: QueryWorkflowState) -> dict:
        try:
            graph_data = state["graph_data"]
            communities = [community for community in graph_data.communities if community.summary]
            ids = [community.id for community in communities]
            limit = min(state.get("top_k", 5), settings.GLOBAL_TOP_K, len(ids))
            selected_ids = await self._rank_named_vectors(
                state["query"],
                ids,
                graph_data.community_embeddings,
                [community.summary or "" for community in communities],
                limit,
                use_stored=self._stored_embeddings_compatible(graph_data),
            )
            by_id = {community.id: community for community in communities}
            ranked = [
                {"id": community_id, "summary": by_id[community_id].summary, "size": by_id[community_id].size}
                for community_id in selected_ids
            ]
            return {
                "relevant_context": {"community_summaries": ranked},
                "current_step": "context_retrieved",
            }
        except Exception as exc:
            return self._failure(exc, "global_search")

    async def basic_search(self, state: QueryWorkflowState) -> dict:
        try:
            graph_data = state["graph_data"]
            chunks = graph_data.text_chunks
            if not chunks:
                return {"relevant_context": {"text_units": []}, "current_step": "context_retrieved"}
            query_vector = await embedding_service.embed_text(state["query"])
            stored_usable = self._stored_embeddings_compatible(graph_data) and all(
                chunk.id in graph_data.chunk_embeddings
                and len(graph_data.chunk_embeddings[chunk.id]) == len(query_vector)
                for chunk in chunks
            )
            if stored_usable:
                vectors = [np.asarray(graph_data.chunk_embeddings[chunk.id], dtype=np.float32) for chunk in chunks]
            else:
                vectors = await embedding_service.embed_texts([chunk.text for chunk in chunks])
            indices = embedding_service.find_most_similar(
                query_vector, vectors, top_k=min(state.get("top_k", 5), settings.MAX_QUERY_TOP_K)
            )
            selected = [chunks[index] for index in indices]
            return {
                "relevant_context": {"text_units": [chunk.model_dump() for chunk in selected]},
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
            answer = await llm_service.generate(prompt, task="local_query_answer")
            return {"answer": answer.strip(), "current_step": "completed"}
        except Exception as exc:
            return self._failure(exc, "generate_local_answer")

    async def generate_global_answer(self, state: QueryWorkflowState) -> dict:
        try:
            prompt = create_global_search_prompt(
                state["query"], state["relevant_context"].get("community_summaries", [])
            )
            answer = await llm_service.generate(prompt, task="global_query_answer")
            return {"answer": answer.strip(), "current_step": "completed"}
        except Exception as exc:
            return self._failure(exc, "generate_global_answer")

    async def generate_basic_answer(self, state: QueryWorkflowState) -> dict:
        try:
            prompt = create_basic_search_prompt(
                state["query"], state["relevant_context"].get("text_units", [])
            )
            answer = await llm_service.generate(prompt, task="basic_query_answer")
            return {"answer": answer.strip(), "current_step": "completed"}
        except Exception as exc:
            return self._failure(exc, "generate_basic_answer")

    async def run(self, query: str, index_id: str, mode: QueryMode = "local", top_k: int = 5) -> dict:
        initial: QueryWorkflowState = {
            "query": query,
            "index_id": index_id,
            "mode": mode,
            "top_k": top_k,
            "relevant_context": {},
            "answer": "",
            "current_step": "init",
            "error": None,
        }
        final_state = await self.graph.ainvoke(initial)
        log.info("Query workflow {}:{} finished at {}", index_id, mode, final_state.get("current_step"))
        return final_state


query_workflow = QueryWorkflow()
