"""Pydantic перевіряє HTTP-межу; пошук усередині працює з dataclass."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SearchParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    q: str = Field(min_length=1, max_length=300)
    k: int = Field(default=10, ge=1, le=100)
    scorer: Literal["bm25", "tfidf"] = "bm25"
    mode: Literal["keyword", "semantic", "hybrid"] = "keyword"
    page: int = Field(default=1, ge=1, le=10000)

    @field_validator("q")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Запит не може бути порожнім")
        return value.strip()


class ResultOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    doc_id: int
    title: str
    score: float
    snippet: str


class SearchResponse(BaseModel):
    query: str
    total: int
    took_ms: float
    page: int
    pages: int
    results: list[ResultOut]


class DocumentOut(BaseModel):
    doc_id: int
    title: str
    path: str
    text: str


class StatsOut(BaseModel):
    documents: int
    vocabulary: int
    tokens: int
    index_bytes: int
    uptime_seconds: float


class HealthOut(BaseModel):
    status: Literal["ok"] = "ok"
