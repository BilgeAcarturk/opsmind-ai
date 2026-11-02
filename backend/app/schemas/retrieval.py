from pydantic import BaseModel, ConfigDict, Field

from app.schemas.chunk import DocumentChunk


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk: DocumentChunk
    score: float = Field(allow_inf_nan=False)
    rank: int = Field(ge=1)
