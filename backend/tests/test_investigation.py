from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.schemas.investigation import (
    InvestigationRequest,
    InvestigationResult,
    InvestigationStatus,
    RootCauseHypothesis,
)


def test_valid_investigation_request():
    request = InvestigationRequest(
        incident_id="INC-001",
        question="What caused the payment service failure?",
    )

    assert request.incident_id == "INC-001"


def test_short_question_is_rejected():
    with pytest.raises(ValidationError):
        InvestigationRequest(
            incident_id="INC-001",
            question="Why?",
        )


def test_completed_investigation():
    result = InvestigationResult(
        investigation_id="INV-001",
        incident_id="INC-001",
        status=InvestigationStatus.COMPLETED,
        created_at=datetime.now(UTC),
        summary="Payment provider timeout detected.",
        hypotheses=[
            RootCauseHypothesis(
                description="External dependency timeout",
                supporting_evidence_ids=["EV-001"],
                next_steps=["Check provider latency"],
            )
        ],
    )

    assert len(result.hypotheses) == 1


def test_completed_result_requires_summary():
    with pytest.raises(ValidationError):
        InvestigationResult(
            investigation_id="INV-002",
            incident_id="INC-001",
            status="completed",
            created_at=datetime.now(UTC),
        )


def test_insufficient_evidence_cannot_have_hypotheses():
    with pytest.raises(ValidationError):
        InvestigationResult(
            investigation_id="INV-003",
            incident_id="INC-001",
            status="completed",
            created_at=datetime.now(UTC),
            summary="Not enough evidence.",
            insufficient_evidence=True,
            hypotheses=[
                RootCauseHypothesis(
                    description="Possible timeout",
                    supporting_evidence_ids=["EV-001"],
                )
            ],
        )


def test_naive_timestamp_is_rejected():
    with pytest.raises(ValidationError):
        InvestigationResult(
            investigation_id="INV-004",
            incident_id="INC-001",
            status="pending",
            created_at=datetime(2026, 10, 25, 14, 0),
        )
