from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class DocumentType(str, Enum):
    RUNBOOK = "runbook"
    POSTMORTEM = "postmortem"


class KnowledgeDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    type: DocumentType
    title: str = Field(min_length=3)
    service_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    content: str = Field(min_length=20)
    version: int = Field(ge=1)
