"""LangGraph indexing workflow for building a portable GraphRAG index."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from config import settings
from core.chunking import text_chunker
from core.claims import claim_extractor
from core.community import community_detector
from core.description_summarization import description_summarizer
from core.entity_resolution import entity_resolver
from core.extraction import entity_relation_extractor
from core.graph_builder import graph_builder
from core.pruning import graph_pruner
from core.summarization import community_summarizer
from models.graph import KnowledgeGraph
from models.schemas import Claim, ExtractionResult, GraphData, TextChunk
from services.embedding_service import embedding_service
from services.storage_service import storage_service
from utils.logger import log, log_exception


class IndexingWorkflowState(TypedDict, total=False):
    index_id: str
    documents: List[str]
    metadata: Dict[str, Any]
    chunks: List[TextChunk]
    extraction_results: List[ExtractionResult]
    claims: List[Claim]
    quality_report: Dict[str, Any]
    kg_object: KnowledgeGraph
    graph_data: GraphData
    current_step: str
    error: str | None


class IndexingWorkflow:
    def __init__(self):
        self.graph = self._build_graph()

    @staticmethod
    def _route(state: IndexingWorkflowState) -> Literal["continue", "end"]:
        return "end" if state.get("error") else "continue"

    @staticmethod
    def _failure(exc: Exception, context: str) -> dict:
        log_exception(exc, context)
        return {"current_step": "error", "error": str(exc)}

    @staticmethod
    def _quality_update(
        state: IndexingWorkflowState,
        key: str,
        value: Any,
    ) -> Dict[str, Any]:
        report = dict(state.get("quality_report", {}))
        report[key] = value
        return report

    def _build_graph(self):
        workflow = StateGraph(IndexingWorkflowState)
        workflow.add_node("chunk_documents", self.chunk_documents)
        workflow.add_node("extract_entities", self.extract_entities)
        workflow.add_node("resolve_entities", self.resolve_entities)
        workflow.add_node("build_graph", self.build_graph_node)
        workflow.add_node("summarize_descriptions", self.summarize_descriptions)
        workflow.add_node("prune_graph", self.prune_graph)
        workflow.add_node("extract_claims", self.extract_claims)
        workflow.add_node("detect_communities", self.detect_communities)
        workflow.add_node("generate_summaries", self.generate_summaries)
        workflow.add_node("build_retrieval_index", self.build_retrieval_index)
        workflow.add_node("save_graph", self.save_graph)

        workflow.add_edge(START, "chunk_documents")
        steps = [
            ("chunk_documents", "extract_entities"),
            ("extract_entities", "resolve_entities"),
            ("resolve_entities", "build_graph"),
            ("build_graph", "summarize_descriptions"),
            ("summarize_descriptions", "prune_graph"),
            ("prune_graph", "extract_claims"),
            ("extract_claims", "detect_communities"),
            ("detect_communities", "generate_summaries"),
            ("generate_summaries", "build_retrieval_index"),
            ("build_retrieval_index", "save_graph"),
        ]
        for source, destination in steps:
            workflow.add_conditional_edges(
                source,
                self._route,
                {"continue": destination, "end": END},
            )
        workflow.add_edge("save_graph", END)
        return workflow.compile()

    async def chunk_documents(self, state: IndexingWorkflowState) -> dict:
        try:
            doc_ids = [
                f"{state['index_id']}_doc_{index}"
                for index in range(len(state["documents"]))
            ]
            chunks = text_chunker.chunk_documents(
                state["documents"],
                doc_ids,
            )
            return {
                "chunks": chunks,
                "current_step": "chunked",
                "error": None,
            }
        except Exception as exc:
            return self._failure(exc, "chunk_documents")

    async def extract_entities(self, state: IndexingWorkflowState) -> dict:
        try:
            results = await entity_relation_extractor.extract_from_chunks(
                state.get("chunks", [])
            )
            return {
                "extraction_results": results,
                "current_step": "extracted",
            }
        except Exception as exc:
            return self._failure(exc, "extract_entities")

    async def resolve_entities(self, state: IndexingWorkflowState) -> dict:
        try:
            results, report = entity_resolver.resolve(
                state.get("extraction_results", [])
            )
            return {
                "extraction_results": results,
                "quality_report": self._quality_update(
                    state,
                    "entity_resolution",
                    report,
                ),
                "current_step": "entities_resolved",
            }
        except Exception as exc:
            return self._failure(exc, "resolve_entities")

    async def build_graph_node(self, state: IndexingWorkflowState) -> dict:
        try:
            kg = graph_builder.build_graph(
                state.get("extraction_results", [])
            )
            return {
                "kg_object": kg,
                "graph_data": kg.to_graph_data(),
                "current_step": "graph_built",
            }
        except Exception as exc:
            return self._failure(exc, "build_graph")

    async def summarize_descriptions(self, state: IndexingWorkflowState) -> dict:
        try:
            kg = await description_summarizer.summarize_graph(
                state["kg_object"]
            )
            return {
                "kg_object": kg,
                "graph_data": kg.to_graph_data(),
                "quality_report": self._quality_update(
                    state,
                    "description_summarization",
                    {
                        "enabled": settings.DESCRIPTION_SUMMARIZATION_ENABLED,
                        "entities": len(kg.entities),
                        "relations": len(kg.relations),
                    },
                ),
                "current_step": "descriptions_summarized",
            }
        except Exception as exc:
            return self._failure(exc, "summarize_descriptions")

    async def prune_graph(self, state: IndexingWorkflowState) -> dict:
        try:
            kg, report = graph_pruner.prune(state["kg_object"])
            return {
                "kg_object": kg,
                "graph_data": kg.to_graph_data(),
                "quality_report": self._quality_update(
                    state,
                    "graph_pruning",
                    report,
                ),
                "current_step": "graph_pruned",
            }
        except Exception as exc:
            return self._failure(exc, "prune_graph")

    async def extract_claims(self, state: IndexingWorkflowState) -> dict:
        try:
            claims = await claim_extractor.extract(
                state.get("chunks", []),
                state["kg_object"],
            )
            return {
                "claims": claims,
                "quality_report": self._quality_update(
                    state,
                    "claims",
                    {
                        "enabled": settings.CLAIM_EXTRACTION_ENABLED,
                        "count": len(claims),
                    },
                ),
                "current_step": "claims_extracted",
            }
        except Exception as exc:
            return self._failure(exc, "extract_claims")

    async def detect_communities(self, state: IndexingWorkflowState) -> dict:
        try:
            kg = state["kg_object"]
            communities = community_detector.detect_communities(kg)
            kg.set_communities(communities)
            return {
                "kg_object": kg,
                "graph_data": kg.to_graph_data(),
                "current_step": "communities_detected",
            }
        except Exception as exc:
            return self._failure(exc, "detect_communities")

    async def generate_summaries(self, state: IndexingWorkflowState) -> dict:
        try:
            kg = state["kg_object"]
            communities = await community_summarizer.summarize_communities(
                kg.communities,
                kg,
            )
            kg.set_communities(communities)
            return {
                "kg_object": kg,
                "graph_data": kg.to_graph_data(),
                "current_step": "summaries_generated",
            }
        except Exception as exc:
            return self._failure(exc, "generate_summaries")

    async def build_retrieval_index(self, state: IndexingWorkflowState) -> dict:
        try:
            kg = state["kg_object"]
            chunks = state.get("chunks", [])
            graph_data = kg.to_graph_data()
            graph_data.text_chunks = chunks
            graph_data.covariates = state.get("claims", [])
            graph_data.quality_report = state.get("quality_report", {})

            entity_names = sorted(kg.entities)
            entity_texts = [
                f"{name}: {kg.entities[name].description}"
                for name in entity_names
            ]
            entity_vectors = await embedding_service.embed_texts(entity_texts)
            graph_data.entity_embeddings = {
                name: vector.tolist()
                for name, vector in zip(entity_names, entity_vectors)
            }

            chunk_vectors = await embedding_service.embed_texts(
                [chunk.text for chunk in chunks]
            )
            graph_data.chunk_embeddings = {
                chunk.id: vector.tolist()
                for chunk, vector in zip(chunks, chunk_vectors)
            }

            summarized = [
                community
                for community in kg.communities
                if community.summary
            ]
            community_vectors = await embedding_service.embed_texts(
                [community.summary or "" for community in summarized]
            )
            graph_data.community_embeddings = {
                community.id: vector.tolist()
                for community, vector in zip(
                    summarized,
                    community_vectors,
                )
            }

            community_levels = sorted(
                {community.level for community in kg.communities}
            )
            graph_data.metadata.update(
                {
                    "schema_version": settings.INDEX_SCHEMA_VERSION,
                    "embedding_provider": settings.EMBEDDING_PROVIDER,
                    "embedding_model": embedding_service.model,
                    "embedding_dimension": embedding_service.dimension,
                    "community_algorithm": settings.COMMUNITY_ALGORITHM,
                    "community_levels": community_levels,
                    "community_max_cluster_size": (
                        settings.COMMUNITY_MAX_CLUSTER_SIZE
                    ),
                    "claims_enabled": settings.CLAIM_EXTRACTION_ENABLED,
                    "num_claims": len(graph_data.covariates),
                    "source_metadata": state.get("metadata", {}),
                }
            )
            return {
                "graph_data": graph_data,
                "current_step": "retrieval_index_built",
            }
        except Exception as exc:
            return self._failure(exc, "build_retrieval_index")

    async def save_graph(self, state: IndexingWorkflowState) -> dict:
        try:
            storage_service.save_graph(
                state["index_id"],
                state["graph_data"],
            )
            return {"current_step": "completed"}
        except Exception as exc:
            return self._failure(exc, "save_graph")

    async def run(
        self,
        index_id: str,
        documents: List[str],
        metadata: Dict[str, Any] | None = None,
    ) -> dict:
        initial: IndexingWorkflowState = {
            "index_id": index_id,
            "documents": documents,
            "metadata": metadata or {},
            "chunks": [],
            "extraction_results": [],
            "claims": [],
            "quality_report": {},
            "current_step": "init",
            "error": None,
        }
        final_state = await self.graph.ainvoke(initial)
        log.info(
            "Indexing workflow {} finished at {}",
            index_id,
            final_state.get("current_step"),
        )
        return final_state


indexing_workflow = IndexingWorkflow()
