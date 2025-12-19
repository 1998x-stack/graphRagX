"""
GraphRAG FastAPI 主应用
使用 Uvicorn 运行
"""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config import settings
from utils.logger import log
from api.index import router as index_router
from api.query import router as query_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    log.info("=" * 50)
    log.info("GraphRAG Application Starting...")
    log.info(f"API Host: {settings.API_HOST}:{settings.API_PORT}")
    log.info(f"LLM Model: {settings.LLM_MODEL}")
    log.info(f"Max Concurrency: {settings.MAX_CONCURRENCY}")
    log.info(f"Output Directory: {settings.OUTPUT_DIR}")
    log.info("=" * 50)
    
    yield
    
    # 关闭时
    log.info("GraphRAG Application Shutting Down...")


# 创建 FastAPI 应用
app = FastAPI(
    title="GraphRAG API",
    description="Knowledge Graph Retrieval Augmented Generation API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(index_router)
app.include_router(query_router)


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "GraphRAG API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "index": "/api/v1/index",
            "query": "/api/v1/query",
            "health": "/api/v1/health"
        }
    }


@app.get("/api/v1/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "GraphRAG",
        "config": {
            "llm_model": settings.LLM_MODEL,
            "embedding_model": settings.EMBEDDING_MODEL,
            "max_concurrency": settings.MAX_CONCURRENCY,
            "chunk_size": settings.CHUNK_SIZE
        }
    }


def main():
    """主入口函数"""
    try:
        # 使用 Uvicorn 运行
        uvicorn.run(
            "main:app",
            host=settings.API_HOST,
            port=settings.API_PORT,
            reload=settings.API_RELOAD,
            log_level="info"
        )
    except Exception as e:
        log.exception(f"Failed to start application: {e}")
        raise


if __name__ == "__main__":
    main()