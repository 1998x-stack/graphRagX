"""Typed evaluation dataset and result schemas."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from models.schemas import QueryMode


class EvaluationCase(BaseModel):
    id: str = Field(..., min_length=1)
    index_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    modes: list[QueryMode] = Field(
        default_factory=lambda: ["basic", "local", "global", "drift"]
    )
    expected_source_ids: list[str] = Field(default_factory=list)
    expected_answer_terms: list[str] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=50)
    community_level: Optional[int] = Field(default=None, ge=0, le=16)


class EvaluationResult(BaseModel):
    case_id: str
    mode: QueryMode
    success: bool
    answer: str = ""
    source_ids: list[str] = Field(default_factory=list)
    source_recall: float = Field(default=0.0, ge=0.0, le=1.0)
    answer_term_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    error: Optional[str] = None
    drift_actions: Optional[int] = None
