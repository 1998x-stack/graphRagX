"""Application configuration for graphRagX."""
from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = False
    API_CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173"

    LLM_PROVIDER: Literal["mock", "openai"] = "mock"
    EMBEDDING_PROVIDER: Literal["hash", "openai"] = "hash"

    LLM_MODEL: str = "gpt-4.1-mini"
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = ""
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 4000

    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BASE_URL: str = ""
    EMBEDDING_DIMENSION: int = 256
    EMBEDDING_BATCH_SIZE: int = 64

    MAX_CONCURRENCY: int = 4
    MAX_RETRY: int = 3
    RETRY_DELAY: float = 1.0

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 160

    MAX_ENTITIES_PER_CHUNK: int = 20
    MAX_RELATIONS_PER_CHUNK: int = 30

    COMMUNITY_ALGORITHM: Literal["louvain", "leiden"] = "louvain"
    COMMUNITY_MAX_LEVELS: int = 3
    COMMUNITY_MAX_CLUSTER_SIZE: int = 20
    COMMUNITY_RESOLUTION_MULTIPLIER: float = 1.5
    LEIDEN_RESOLUTION: float = 1.0
    LEIDEN_MAX_ITERATIONS: int = 10
    LEIDEN_SEED: int = 42

    RETRIEVAL_NEIGHBOR_DEPTH: int = 1
    GLOBAL_TOP_K: int = 24
    GLOBAL_COMMUNITY_LEVEL: int = 1
    GLOBAL_MAX_DATA_TOKENS: int = 8000
    GLOBAL_MAP_BATCH_TOKENS: int = 2000
    GLOBAL_REDUCE_DATA_TOKENS: int = 4000
    LOCAL_MAX_DATA_TOKENS: int = 4000
    MAX_QUERY_TOP_K: int = 50
    MAX_CONTEXT_CHUNKS: int = 12

    MAX_DOCUMENTS: int = 200
    MAX_DOCUMENT_CHARS: int = 500_000
    MAX_QUERY_CHARS: int = 8_000

    ENABLE_STREAM: bool = False
    STREAM_LOG_ENABLED: bool = False
    STORE_LLM_PAYLOADS: bool = False

    OUTPUT_DIR: Path = Path("./outputs")
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Path = Path("./logs/graphrag.log")
    LOG_ROTATION: str = "100 MB"
    LOG_RETENTION: str = "10 days"
    INDEX_SCHEMA_VERSION: str = "2.1"

    @property
    def LLM_LOGS_DIR(self) -> Path:
        return self.OUTPUT_DIR / "llm_logs"

    @property
    def GRAPHS_DIR(self) -> Path:
        return self.OUTPUT_DIR / "graphs"

    @property
    def COMMUNITIES_DIR(self) -> Path:
        return self.OUTPUT_DIR / "communities"

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.API_CORS_ORIGINS.split(",") if item.strip()]

    @model_validator(mode="after")
    def validate_runtime_settings(self) -> "Settings":
        if self.CHUNK_SIZE <= 0:
            raise ValueError("CHUNK_SIZE must be > 0")
        if not 0 <= self.CHUNK_OVERLAP < self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must satisfy 0 <= overlap < CHUNK_SIZE")
        if self.MAX_CONCURRENCY <= 0:
            raise ValueError("MAX_CONCURRENCY must be > 0")
        if self.EMBEDDING_DIMENSION <= 0:
            raise ValueError("EMBEDDING_DIMENSION must be > 0")
        if self.EMBEDDING_BATCH_SIZE <= 0:
            raise ValueError("EMBEDDING_BATCH_SIZE must be > 0")
        if self.MAX_QUERY_TOP_K <= 0:
            raise ValueError("MAX_QUERY_TOP_K must be > 0")
        if self.COMMUNITY_MAX_LEVELS <= 0:
            raise ValueError("COMMUNITY_MAX_LEVELS must be > 0")
        if self.COMMUNITY_MAX_CLUSTER_SIZE <= 0:
            raise ValueError("COMMUNITY_MAX_CLUSTER_SIZE must be > 0")
        if self.COMMUNITY_RESOLUTION_MULTIPLIER <= 1.0:
            raise ValueError("COMMUNITY_RESOLUTION_MULTIPLIER must be > 1")
        for name in (
            "GLOBAL_MAX_DATA_TOKENS",
            "GLOBAL_MAP_BATCH_TOKENS",
            "GLOBAL_REDUCE_DATA_TOKENS",
            "LOCAL_MAX_DATA_TOKENS",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        return self

    def ensure_directories(self) -> None:
        for path in (
            self.OUTPUT_DIR,
            self.LLM_LOGS_DIR,
            self.GRAPHS_DIR,
            self.COMMUNITIES_DIR,
            self.LOG_FILE.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
