from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.schemas.incident import Incident


def test_valid_incident():
    incident = Incident(
        id="INC-001",
        service_id="payment-service",
        title="Payment service timeout",
        severity="SEV2",
        status="investigating",
        started_at=datetime(2026, 10, 25, 14, 0, tzinfo=UTC),
    )

    assert incident.id == "INC-001"
    assert incident.severity.value == "SEV2"
    assert incident.symptoms == []


def test_invalid_severity():
    with pytest.raises(ValidationError):
        Incident(
            id="INC-002",
            service_id="payment-service",
            title="Payment service failure",
            severity="INVALID",
            status="open",
            started_at=datetime.now(UTC),
        )


def test_invalid_time_range():
    with pytest.raises(ValidationError):
        Incident(
            id="INC-003",
            service_id="payment-service",
            title="Payment service failure",
            severity="SEV2",
            status="resolved",
            started_at="2026-10-25T15:00:00Z",
            ended_at="2026-10-25T14:00:00Z",
        )


def test_missing_timezone():
    with pytest.raises(ValidationError):
        Incident(
            id="INC-004",
            service_id="payment-service",
            title="Payment service failure",
            severity="SEV2",
            status="open",
            started_at="2026-10-25T14:00:00",
        )
