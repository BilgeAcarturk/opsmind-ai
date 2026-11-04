from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.retrieval import SearchResult


class RetrievalSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=3,
        max_length=1000,
    )
    top_k: int = Field(default=5, ge=1, le=20)
    service_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        cleaned = value.strip()

        if len(cleaned) < 3:
            raise ValueError("Query must contain at least 3 characters")

        return cleaned


class RetrievalSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    method: str
    total_results: int
    results: list[SearchResult]
