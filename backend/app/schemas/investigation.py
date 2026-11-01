from datetime import datetime
from enum import Enum
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class InvestigationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class InvestigationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    incident_id: str = Field(min_length=1)
    question: str = Field(min_length=10, max_length=1000)


class RootCauseHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)
    supporting_evidence_ids: list[str] = Field(min_length=1)
    next_steps: list[str] = Field(default_factory=list)


class InvestigationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    investigation_id: str = Field(min_length=1)
    incident_id: str = Field(min_length=1)
    status: InvestigationStatus
    created_at: datetime
    summary: str | None = None
    hypotheses: list[RootCauseHypothesis] = Field(default_factory=list)
    insufficient_evidence: bool = False

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.created_at.utcoffset() is None:
            raise ValueError("created_at must include a timezone")

        if self.status == InvestigationStatus.COMPLETED and not self.summary:
            raise ValueError("Completed investigation requires a summary")

        if self.insufficient_evidence and self.hypotheses:
            raise ValueError("Insufficient evidence cannot have hypotheses")

        return self
