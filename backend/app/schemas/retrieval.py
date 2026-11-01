from pydantic import BaseModel, ConfigDict, Field

from app.schemas.chunk import DocumentChunk


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk: DocumentChunk
    score: float = Field(ge=-1.0, le=1.0)
    rank: int = Field(ge=1)
