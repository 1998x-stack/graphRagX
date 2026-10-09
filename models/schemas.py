"""Validated domain and API schemas."""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


INDEX_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
QueryMode = Literal["local", "global", "basic", "drift"]


class Entity(BaseModel):
    name: str = Field(..., min_length=1)
    type: str = Field(..., min_length=1)
    description: str = Field(default="")
    aliases: List[str] = Field(default_factory=list)
    mention_count: int = Field(default=1, ge=1)
    description_mentions: List[str] = Field(default_factory=list)
    source_chunk_ids: List[str] = Field(default_factory=list)

    def __hash__(self) -> int:
        return hash((self.name, self.type))


class Relation(BaseModel):
    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    relation_type: str = Field(..., min_length=1)
    description: str = Field(default="")
    mention_count: int = Field(default=1, ge=1)
    description_mentions: List[str] = Field(default_factory=list)
    weight: float = Field(default=1.0, gt=0)
    source_chunk_ids: List[str] = Field(default_factory=list)

    def __hash__(self) -> int:
        return hash((self.source, self.target, self.relation_type))


class Claim(BaseModel):
    id: str = Field(..., min_length=1)
    covariate_type: Literal["claim"] = "claim"
    type: str = Field(default="FACT")
    description: str = Field(..., min_length=1)
    subject_id: str = Field(..., min_length=1)
    object_id: Optional[str] = None
    status: Literal["TRUE", "FALSE", "SUSPECTED", "UNKNOWN"] = "UNKNOWN"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    source_text: str = Field(default="")
    text_unit_id: str = Field(..., min_length=1)


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
    covariates: List[Claim] = Field(default_factory=list)
    text_chunks: List[TextChunk] = Field(default_factory=list)
    entity_embeddings: Dict[str, List[float]] = Field(default_factory=dict)
    chunk_embeddings: Dict[str, List[float]] = Field(default_factory=dict)
    community_embeddings: Dict[str, List[float]] = Field(default_factory=dict)
    quality_report: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DriftFollowUp(BaseModel):
    id: str
    question: str = Field(..., min_length=1)
    score: float = Field(default=0.0, ge=0.0, le=100.0)
    depth: int = Field(default=1, ge=1)
    parent_id: Optional[str] = None


class DriftEvidence(BaseModel):
    id: str
    question: str
    answer: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    depth: int = Field(default=1, ge=1)
    parent_id: Optional[str] = None
    source_ids: List[str] = Field(default_factory=list)


class DriftTrace(BaseModel):
    primer_answer: str = ""
    primer_reports: List[str] = Field(default_factory=list)
    follow_ups: List[DriftFollowUp] = Field(default_factory=list)
    evidence: List[DriftEvidence] = Field(default_factory=list)
    visited_questions: List[str] = Field(default_factory=list)
    actions_executed: int = 0
    max_depth_reached: int = 0
    termination_reason: str = ""


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
    mode: QueryMode = "local"
    top_k: int = Field(default=5, ge=1, le=50)
    community_level: Optional[int] = Field(default=None, ge=0, le=16)


class QueryResponse(BaseModel):
    status: Literal["success", "failed"]
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    mode: QueryMode
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
    mode: QueryMode
    top_k: int = 5
    community_level: Optional[int] = None
    graph_data: Optional[GraphData] = None
    relevant_context: Dict[str, Any] = Field(default_factory=dict)
    drift_trace: Optional[DriftTrace] = None
    answer: str = ""
    current_step: str = "init"
    error: Optional[str] = None
