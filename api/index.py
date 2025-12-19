"""
索引 API 路由
提供文档索引的 REST 接口
"""
import time
from fastapi import APIRouter, HTTPException
from models.schemas import IndexRequest, IndexResponse
from workflows.indexing_workflow import indexing_workflow
from utils.logger import log, log_exception


router = APIRouter(prefix="/api/v1", tags=["index"])


@router.post("/index", response_model=IndexResponse)
async def create_index(request: IndexRequest):
    """
    创建索引
    
    流程:
    1. 文本分块
    2. 实体/关系提取
    3. 图谱构建
    4. 社区检测
    5. 社区摘要生成
    6. 保存到存储
    
    Args:
        request: 索引请求
        
    Returns:
        索引响应
    """
    start_time = time.time()
    
    try:
        log.info(f"Received index request: index_id={request.index_id}, docs={len(request.documents)}")
        
        # 验证输入
        if not request.documents:
            raise HTTPException(status_code=400, detail="No documents provided")
        
        if not request.index_id:
            raise HTTPException(status_code=400, detail="No index_id provided")
        
        # 运行索引工作流
        final_state = await indexing_workflow.run(
            index_id=request.index_id,
            documents=request.documents
        )
        
        # 检查是否有错误
        if final_state.get("error"):
            log.error(f"Indexing failed: {final_state['error']}")
            return IndexResponse(
                status="failed",
                index_id=request.index_id,
                num_documents=len(request.documents),
                num_chunks=len(final_state.get("chunks", [])),
                num_entities=0,
                num_relations=0,
                num_communities=0,
                processing_time=time.time() - start_time,
                error=final_state["error"]
            )
        
        # 提取统计信息
        graph_data = final_state.get("graph_data")
        
        response = IndexResponse(
            status="success",
            index_id=request.index_id,
            num_documents=len(request.documents),
            num_chunks=len(final_state.get("chunks", [])),
            num_entities=len(graph_data.entities) if graph_data else 0,
            num_relations=len(graph_data.relations) if graph_data else 0,
            num_communities=len(graph_data.communities) if graph_data else 0,
            processing_time=time.time() - start_time
        )
        
        log.info(f"Index created successfully: {request.index_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        log_exception(e, "create_index")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/indexes")
async def list_indexes():
    """
    列出所有索引
    
    Returns:
        索引ID列表
    """
    try:
        from services.storage_service import storage_service
        indexes = storage_service.list_indexes()
        
        log.info(f"Listed {len(indexes)} indexes")
        return {
            "status": "success",
            "indexes": indexes,
            "count": len(indexes)
        }
        
    except Exception as e:
        log_exception(e, "list_indexes")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/index/{index_id}")
async def delete_index(index_id: str):
    """
    删除索引（TODO: 实现）
    
    Args:
        index_id: 索引ID
        
    Returns:
        删除结果
    """
    # TODO: 实现删除逻辑
    log.warning(f"Delete index not implemented: {index_id}")
    return {
        "status": "not_implemented",
        "message": "Delete functionality not yet implemented"
    }