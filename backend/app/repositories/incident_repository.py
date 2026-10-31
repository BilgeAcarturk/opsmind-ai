from typing import Protocol

from app.schemas.incident import Incident


class IncidentRepository(Protocol):
    def get_all(self) -> list[Incident]: ...

    def get_by_id(self, incident_id: str) -> Incident | None: ...


class InMemoryIncidentRepository:
    def __init__(self):
        incident = Incident(
            id="INC-001",
            service_id="payment-service",
            title="Payment API error rate increased",
            severity="SEV2",
            status="investigating",
            started_at="2026-10-25T14:00:00Z",
            symptoms=[
                "HTTP 5xx error rate increased",
                "Payment requests timing out",
            ],
        )
        self._incidents = {incident.id: incident}

    def get_all(self) -> list[Incident]:
        return [incident.model_copy(deep=True) for incident in self._incidents.values()]

    def get_by_id(self, incident_id: str) -> Incident | None:
        incident = self._incidents.get(incident_id)
        return incident.model_copy(deep=True) if incident is not None else None
