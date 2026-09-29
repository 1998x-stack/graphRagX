"""Index management API."""
from time import perf_counter

from fastapi import APIRouter, HTTPException, status

from config import settings
from models.schemas import IndexRequest, IndexResponse
from services.storage_service import storage_service
from utils.logger import log, log_exception
from workflows.indexing_workflow import indexing_workflow

router = APIRouter(prefix="/api/v1", tags=["index"])


@router.post("/index", response_model=IndexResponse)
async def create_index(request: IndexRequest):
    started = perf_counter()
    if len(request.documents) > settings.MAX_DOCUMENTS:
        raise HTTPException(status_code=413, detail=f"Too many documents; max={settings.MAX_DOCUMENTS}")
    total_chars = sum(len(document) for document in request.documents)
    if total_chars > settings.MAX_DOCUMENT_CHARS:
        raise HTTPException(status_code=413, detail=f"Document payload too large; max_chars={settings.MAX_DOCUMENT_CHARS}")
    if storage_service.index_exists(request.index_id) and not request.overwrite:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Index already exists; set overwrite=true to replace it")

    try:
        final_state = await indexing_workflow.run(
            request.index_id,
            request.documents,
            metadata=request.metadata,
        )
        graph_data = final_state.get("graph_data")
        if final_state.get("error"):
            return IndexResponse(
                status="failed",
                index_id=request.index_id,
                num_documents=len(request.documents),
                num_chunks=len(final_state.get("chunks", [])),
                num_entities=len(graph_data.entities) if graph_data else 0,
                num_relations=len(graph_data.relations) if graph_data else 0,
                num_communities=len(graph_data.communities) if graph_data else 0,
                processing_time=perf_counter() - started,
                error=final_state["error"],
            )
        return IndexResponse(
            status="success",
            index_id=request.index_id,
            num_documents=len(request.documents),
            num_chunks=len(final_state.get("chunks", [])),
            num_entities=len(graph_data.entities) if graph_data else 0,
            num_relations=len(graph_data.relations) if graph_data else 0,
            num_communities=len(graph_data.communities) if graph_data else 0,
            processing_time=perf_counter() - started,
        )
    except HTTPException:
        raise
    except Exception as exc:
        log_exception(exc, "create_index")
        raise HTTPException(status_code=500, detail="Indexing failed") from exc


@router.get("/indexes")
async def list_indexes():
    indexes = storage_service.list_indexes()
    return {"status": "success", "indexes": indexes, "count": len(indexes)}


@router.delete("/index/{index_id}")
async def delete_index(index_id: str):
    try:
        deleted = storage_service.delete_index(index_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Index not found: {index_id}")
    log.info("Deleted index {}", index_id)
    return {"status": "success", "index_id": index_id}
