from datetime import datetime
from enum import Enum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Severity(str, Enum):
    SEV1 = "SEV1"
    SEV2 = "SEV2"
    SEV3 = "SEV3"
    SEV4 = "SEV4"


class IncidentStatus(str, Enum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"


class Incident(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    service_id: str = Field(min_length=1)
    title: str = Field(min_length=3)
    severity: Severity
    status: IncidentStatus
    started_at: datetime
    ended_at: datetime | None = None
    symptoms: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_time_range(self) -> Self:
        if self.started_at.utcoffset() is None:
            raise ValueError("started_at must include a timezone")

        if self.ended_at is not None:
            if self.ended_at.utcoffset() is None:
                raise ValueError("ended_at must include a timezone")
            if self.ended_at < self.started_at:
                raise ValueError("ended_at cannot be earlier than started_at")

        return self
