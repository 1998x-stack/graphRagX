"""Query API for GraphRAG and baseline RAG modes."""
from time import perf_counter

from fastapi import APIRouter, HTTPException

from config import settings
from models.graph import KnowledgeGraph
from models.schemas import QueryRequest, QueryResponse
from services.storage_service import storage_service
from utils.logger import log_exception
from workflows.query_workflow import query_workflow

router = APIRouter(prefix="/api/v1", tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query_graph(request: QueryRequest):
    started = perf_counter()
    if len(request.query) > settings.MAX_QUERY_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Query too large; max_chars={settings.MAX_QUERY_CHARS}",
        )
    if request.top_k > settings.MAX_QUERY_TOP_K:
        raise HTTPException(
            status_code=422,
            detail=f"top_k must be <= {settings.MAX_QUERY_TOP_K}",
        )
    if not storage_service.index_exists(request.index_id):
        raise HTTPException(
            status_code=404,
            detail=f"Index not found: {request.index_id}",
        )

    try:
        final_state = await query_workflow.run(
            query=request.query,
            index_id=request.index_id,
            mode=request.mode,
            top_k=request.top_k,
            community_level=request.community_level,
        )
        if final_state.get("error"):
            return QueryResponse(
                status="failed",
                answer="",
                sources=[],
                mode=request.mode,
                processing_time=perf_counter() - started,
                error=final_state["error"],
            )

        context = final_state.get("relevant_context", {})
        if request.mode == "local":
            sources = [
                {
                    "type": "text_unit",
                    "id": item.get("id"),
                    "doc_id": item.get("doc_id"),
                    "text": item.get("text", "")[:240],
                }
                for item in context.get("text_units", [])
            ]
            sources.extend(
                {
                    "type": "entity",
                    "name": item.get("name"),
                    "description": item.get("description", "")[:200],
                }
                for item in context.get("entities", [])[: request.top_k]
            )
            sources.extend(
                {
                    "type": "claim",
                    "id": item.get("id"),
                    "status": item.get("status"),
                    "subject_id": item.get("subject_id"),
                    "object_id": item.get("object_id"),
                    "description": item.get("description", "")[:240],
                }
                for item in context.get("claims", [])
            )
        elif request.mode == "global":
            sources = [
                {
                    "type": "community",
                    "id": item.get("id"),
                    "level": item.get("level"),
                    "parent_id": item.get("parent_id"),
                    "relevance": item.get("relevance"),
                    "summary": item.get("summary", "")[:240],
                }
                for item in context.get("community_summaries", [])
            ]
        elif request.mode == "drift":
            sources = [
                {
                    "type": "community",
                    "id": item.get("id"),
                    "level": item.get("level"),
                    "relevance": item.get("relevance"),
                    "summary": item.get("summary", "")[:200],
                }
                for item in context.get("community_summaries", [])
            ]
            drift = context.get("drift", {})
            sources.extend(
                {
                    "type": "drift_evidence",
                    "id": item.get("id"),
                    "depth": item.get("depth"),
                    "confidence": item.get("confidence"),
                    "question": item.get("question"),
                    "source_ids": item.get("source_ids", []),
                    "answer": item.get("answer", "")[:240],
                }
                for item in drift.get("evidence", [])
            )
        else:
            sources = [
                {
                    "type": "text_unit",
                    "id": item.get("id"),
                    "doc_id": item.get("doc_id"),
                    "text": item.get("text", "")[:240],
                }
                for item in context.get("text_units", [])
            ]

        return QueryResponse(
            status="success",
            answer=final_state.get("answer", ""),
            sources=sources,
            mode=request.mode,
            processing_time=perf_counter() - started,
        )
    except Exception as exc:
        log_exception(exc, "query_graph")
        raise HTTPException(status_code=500, detail="Query failed") from exc


@router.get("/query/stats/{index_id}")
async def get_query_stats(index_id: str):
    try:
        graph_data = storage_service.load_graph(index_id)
        stats = KnowledgeGraph.from_graph_data(graph_data).get_statistics()
        stats.update(
            {
                "index_id": index_id,
                "num_text_chunks": len(graph_data.text_chunks),
                "community_levels": sorted(
                    {community.level for community in graph_data.communities}
                ),
                "has_entity_embeddings": bool(graph_data.entity_embeddings),
                "has_chunk_embeddings": bool(graph_data.chunk_embeddings),
                "has_community_embeddings": bool(
                    graph_data.community_embeddings
                ),
                "num_claims": len(graph_data.covariates),
                "quality_report": graph_data.quality_report,
            }
        )
        return {"status": "success", "stats": stats}
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Index not found: {index_id}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
