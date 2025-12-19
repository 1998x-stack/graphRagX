"""
查询 API 路由
提供知识图谱查询的 REST 接口
"""
import time
from fastapi import APIRouter, HTTPException
from models.schemas import QueryRequest, QueryResponse
from workflows.query_workflow import query_workflow
from utils.logger import log, log_exception


router = APIRouter(prefix="/api/v1", tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query_graph(request: QueryRequest):
    """
    查询知识图谱
    
    支持两种模式:
    - local: 局部搜索（针对特定实体）
    - global: 全局搜索（针对整体主题）
    
    Args:
        request: 查询请求
        
    Returns:
        查询响应
    """
    start_time = time.time()
    
    try:
        log.info(
            f"Received query request: index_id={request.index_id}, "
            f"mode={request.mode}, query={request.query[:50]}..."
        )
        
        # 验证输入
        if not request.query:
            raise HTTPException(status_code=400, detail="No query provided")
        
        if not request.index_id:
            raise HTTPException(status_code=400, detail="No index_id provided")
        
        if request.mode not in ["local", "global"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode: {request.mode}. Must be 'local' or 'global'"
            )
        
        # 运行查询工作流
        final_state = await query_workflow.run(
            query=request.query,
            index_id=request.index_id,
            mode=request.mode,
            top_k=request.top_k
        )
        
        # 检查是否有错误
        if final_state.get("error"):
            log.error(f"Query failed: {final_state['error']}")
            return QueryResponse(
                status="failed",
                answer="",
                sources=[],
                mode=request.mode,
                processing_time=time.time() - start_time,
                error=final_state["error"]
            )
        
        # 构建来源信息
        sources = []
        context = final_state.get("relevant_context", {})
        
        if request.mode == "local":
            # 局部搜索的来源
            for entity in context.get("entities", [])[:request.top_k]:
                sources.append({
                    "type": "entity",
                    "name": entity.get("name"),
                    "description": entity.get("description", "")[:200]  # 截断
                })
        else:
            # 全局搜索的来源
            for comm in context.get("community_summaries", []):
                sources.append({
                    "type": "community",
                    "id": comm.get("id"),
                    "summary": comm.get("summary", "")[:200]  # 截断
                })
        
        response = QueryResponse(
            status="success",
            answer=final_state.get("answer", ""),
            sources=sources,
            mode=request.mode,
            processing_time=time.time() - start_time
        )
        
        log.info(f"Query completed successfully in {response.processing_time:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        log_exception(e, "query_graph")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/query/stats/{index_id}")
async def get_query_stats(index_id: str):
    """
    获取索引统计信息
    
    Args:
        index_id: 索引ID
        
    Returns:
        统计信息
    """
    try:
        from services.storage_service import storage_service
        from models.graph import KnowledgeGraph
        
        # 加载图谱
        graph_data = storage_service.load_graph(index_id)
        kg = KnowledgeGraph.from_graph_data(graph_data)
        
        # 获取统计
        stats = kg.get_statistics()
        stats["index_id"] = index_id
        stats["num_communities"] = len(kg.communities)
        
        log.info(f"Retrieved stats for index: {index_id}")
        return {
            "status": "success",
            "stats": stats
        }
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Index not found: {index_id}")
    except Exception as e:
        log_exception(e, "get_query_stats")
        raise HTTPException(status_code=500, detail=str(e))