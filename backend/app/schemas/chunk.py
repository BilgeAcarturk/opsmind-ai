from pydantic import BaseModel, ConfigDict, Field

from app.schemas.knowledge import DocumentType


class DocumentChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    document_version: int = Field(ge=1)
    chunk_index: int = Field(ge=0)

    content: str = Field(min_length=1)

    document_title: str = Field(min_length=3)
    document_type: DocumentType
    service_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
