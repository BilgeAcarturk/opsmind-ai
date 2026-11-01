from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RetrievalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    relevant_document_ids: list[str] = Field(min_length=1)
    service_id: str | None = None


class RetrievalDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_version: str
    label_level: str
    cases: list[RetrievalCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_dataset(self) -> Self:
        if self.label_level != "document":
            raise ValueError("Only document-level labels are supported")

        query_ids = [case.query_id for case in self.cases]

        if len(query_ids) != len(set(query_ids)):
            raise ValueError("Duplicate query IDs detected")

        for case in self.cases:
            ids = case.relevant_document_ids
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate relevance labels detected")

        return self
