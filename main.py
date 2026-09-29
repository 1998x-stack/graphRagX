"""FastAPI application entry point."""
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.index import router as index_router
from api.query import router as query_router
from config import settings
from utils.logger import log


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    settings.ensure_directories()
    log.info(
        "graphRagX starting | llm_provider={} embedding_provider={} "
        "community_algorithm={} concurrency={}",
        settings.LLM_PROVIDER,
        settings.EMBEDDING_PROVIDER,
        settings.COMMUNITY_ALGORITHM,
        settings.MAX_CONCURRENCY,
    )
    yield
    log.info("graphRagX shutting down")


app = FastAPI(
    title="graphRagX API",
    description=(
        "Knowledge-graph retrieval augmented generation with local, "
        "global map-reduce, and baseline search."
    ),
    version="2.1.0",
    lifespan=lifespan,
)

origins = settings.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials="*" not in origins,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(index_router)
app.include_router(query_router)


@app.get("/")
async def root():
    return {
        "name": "graphRagX API",
        "version": "2.1.0",
        "modes": ["local", "global", "basic"],
        "docs": "/docs",
    }


@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "graphRagX",
        "providers": {
            "llm": settings.LLM_PROVIDER,
            "embedding": settings.EMBEDDING_PROVIDER,
            "community": settings.COMMUNITY_ALGORITHM,
        },
    }


def main():
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
