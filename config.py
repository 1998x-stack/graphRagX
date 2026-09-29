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

    # V2.3 graph quality.
    ENTITY_RESOLUTION_STRATEGY: Literal["none", "normalized_exact"] = "normalized_exact"
    DESCRIPTION_SUMMARIZATION_ENABLED: bool = True
    DESCRIPTION_SUMMARY_MAX_MENTIONS: int = 12

    GRAPH_PRUNING_ENABLED: bool = True
    PRUNE_MIN_NODE_FREQ: int = 1
    PRUNE_MAX_NODE_FREQ_STD: float | None = None
    PRUNE_MIN_NODE_DEGREE: int = 0
    PRUNE_MAX_NODE_DEGREE_STD: float | None = None
    PRUNE_MIN_EDGE_WEIGHT_PCT: float = 0.0
    PRUNE_REMOVE_EGO_NODES: bool = False
    PRUNE_LCC_ONLY: bool = False

    CLAIM_EXTRACTION_ENABLED: bool = False
    CLAIM_DESCRIPTION: str = "Important factual claims involving indexed entities"
    MAX_CLAIMS_PER_CHUNK: int = 10

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

    DRIFT_K_FOLLOWUPS: int = 3
    DRIFT_PRIMER_FOLDS: int = 3
    DRIFT_N_DEPTH: int = 2
    DRIFT_MAX_ACTIONS: int = 12
    DRIFT_COMMUNITY_LEVEL: int = 1
    DRIFT_PRIMER_DATA_TOKENS: int = 5000
    DRIFT_LOCAL_MAX_DATA_TOKENS: int = 2500
    DRIFT_REDUCE_DATA_TOKENS: int = 5000
    DRIFT_LOCAL_TOP_K_ENTITIES: int = 5
    DRIFT_EXPANSION_MIN_CONFIDENCE: float = 0.35

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
    INDEX_SCHEMA_VERSION: str = "2.3"

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
        positive_names = (
            "CHUNK_SIZE",
            "MAX_CONCURRENCY",
            "EMBEDDING_DIMENSION",
            "EMBEDDING_BATCH_SIZE",
            "MAX_QUERY_TOP_K",
            "COMMUNITY_MAX_LEVELS",
            "COMMUNITY_MAX_CLUSTER_SIZE",
            "DESCRIPTION_SUMMARY_MAX_MENTIONS",
            "MAX_CLAIMS_PER_CHUNK",
            "GLOBAL_MAX_DATA_TOKENS",
            "GLOBAL_MAP_BATCH_TOKENS",
            "GLOBAL_REDUCE_DATA_TOKENS",
            "LOCAL_MAX_DATA_TOKENS",
            "DRIFT_PRIMER_DATA_TOKENS",
            "DRIFT_LOCAL_MAX_DATA_TOKENS",
            "DRIFT_REDUCE_DATA_TOKENS",
            "DRIFT_K_FOLLOWUPS",
            "DRIFT_PRIMER_FOLDS",
            "DRIFT_N_DEPTH",
            "DRIFT_MAX_ACTIONS",
            "DRIFT_LOCAL_TOP_K_ENTITIES",
        )
        for name in positive_names:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if not 0 <= self.CHUNK_OVERLAP < self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must satisfy 0 <= overlap < CHUNK_SIZE")
        if self.COMMUNITY_RESOLUTION_MULTIPLIER <= 1.0:
            raise ValueError("COMMUNITY_RESOLUTION_MULTIPLIER must be > 1")
        if self.PRUNE_MIN_NODE_FREQ < 1:
            raise ValueError("PRUNE_MIN_NODE_FREQ must be >= 1")
        if self.PRUNE_MIN_NODE_DEGREE < 0:
            raise ValueError("PRUNE_MIN_NODE_DEGREE must be >= 0")
        if not 0.0 <= self.PRUNE_MIN_EDGE_WEIGHT_PCT <= 100.0:
            raise ValueError("PRUNE_MIN_EDGE_WEIGHT_PCT must be in [0, 100]")
        if self.PRUNE_MAX_NODE_FREQ_STD is not None and self.PRUNE_MAX_NODE_FREQ_STD < 0:
            raise ValueError("PRUNE_MAX_NODE_FREQ_STD must be >= 0")
        if self.PRUNE_MAX_NODE_DEGREE_STD is not None and self.PRUNE_MAX_NODE_DEGREE_STD < 0:
            raise ValueError("PRUNE_MAX_NODE_DEGREE_STD must be >= 0")
        if not 0.0 <= self.DRIFT_EXPANSION_MIN_CONFIDENCE <= 1.0:
            raise ValueError("DRIFT_EXPANSION_MIN_CONFIDENCE must be in [0, 1]")
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
