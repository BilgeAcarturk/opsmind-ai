from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EvidenceType(str, Enum):
    LOG = "log"
    METRIC = "metric"
    TRACE = "trace"
    DOCUMENT = "document"
    DEPLOYMENT = "deployment"


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    incident_id: str = Field(min_length=1)
    service_id: str = Field(min_length=1)
    type: EvidenceType
    source: str = Field(min_length=1)
    observed_at: datetime
    content: str = Field(min_length=1)
    source_record_id: str = Field(min_length=1)

    @field_validator("observed_at")
    @classmethod
    def check_timezone(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("observed_at must include a timezone")
        return value
