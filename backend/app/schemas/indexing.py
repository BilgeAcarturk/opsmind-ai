from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class IndexingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: str = "fixed_word"
    chunk_size: int = Field(default=100, gt=0)
    overlap: int = Field(default=20, ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> Self:
        if self.strategy != "fixed_word":
            raise ValueError("Unsupported chunking strategy")

        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        return self


class IngestionManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    run_id: str
    created_at: datetime
    config: IndexingConfig
    config_hash: str
    dataset_hash: str
    document_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    document_hashes: dict[str, str]
