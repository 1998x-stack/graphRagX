"""Validated domain and API schemas."""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


INDEX_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"


class Entity(BaseModel):
    name: str = Field(..., min_length=1)
    type: str = Field(..., min_length=1)
    description: str = Field(default="")
    source_chunk_ids: List[str] = Field(default_factory=list)

    def __hash__(self) -> int:
        return hash((self.name, self.type))


class Relation(BaseModel):
    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    relation_type: str = Field(..., min_length=1)
    description: str = Field(default="")
    weight: float = Field(default=1.0, gt=0)
    source_chunk_ids: List[str] = Field(default_factory=list)

    def __hash__(self) -> int:
        return hash((self.source, self.target, self.relation_type))


class Community(BaseModel):
    id: str
    level: int = 0
    entities: List[str] = Field(default_factory=list)
    summary: Optional[str] = None
    parent_id: Optional[str] = None
    size: int = 0


class TextChunk(BaseModel):
    id: str
    text: str
    doc_id: str
    chunk_index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    entities: List[Entity] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)
    chunk_id: str
    raw_response: str = ""


class GraphData(BaseModel):
    """Portable index payload persisted by the local storage adapter."""

    entities: Dict[str, Entity] = Field(default_factory=dict)
    relations: List[Relation] = Field(default_factory=list)
    communities: List[Community] = Field(default_factory=list)
    text_chunks: List[TextChunk] = Field(default_factory=list)
    entity_embeddings: Dict[str, List[float]] = Field(default_factory=dict)
    chunk_embeddings: Dict[str, List[float]] = Field(default_factory=dict)
    community_embeddings: Dict[str, List[float]] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IndexRequest(BaseModel):
    documents: List[str] = Field(..., min_length=1)
    index_id: str = Field(..., pattern=INDEX_ID_PATTERN)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    overwrite: bool = False

    @field_validator("documents")
    @classmethod
    def reject_blank_documents(cls, documents: List[str]) -> List[str]:
        if any(not document.strip() for document in documents):
            raise ValueError("documents must not contain blank entries")
        return documents


class IndexResponse(BaseModel):
    status: Literal["success", "failed"]
    index_id: str
    num_documents: int
    num_chunks: int
    num_entities: int
    num_relations: int
    num_communities: int
    processing_time: float
    error: Optional[str] = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    index_id: str = Field(..., pattern=INDEX_ID_PATTERN)
    mode: Literal["local", "global", "basic"] = "local"
    top_k: int = Field(default=5, ge=1, le=50)
    community_level: Optional[int] = Field(default=None, ge=0, le=16)


class QueryResponse(BaseModel):
    status: Literal["success", "failed"]
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    mode: Literal["local", "global", "basic"]
    processing_time: float
    error: Optional[str] = None


class IndexingState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    index_id: str
    documents: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunks: List[TextChunk] = Field(default_factory=list)
    extraction_results: List[ExtractionResult] = Field(default_factory=list)
    graph_data: Optional[GraphData] = None
    current_step: str = "init"
    error: Optional[str] = None


class QueryState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    query: str
    index_id: str
    mode: Literal["local", "global", "basic"]
    top_k: int = 5
    community_level: Optional[int] = None
    graph_data: Optional[GraphData] = None
    relevant_context: Dict[str, Any] = Field(default_factory=dict)
    answer: str = ""
    current_step: str = "init"
    error: Optional[str] = None
