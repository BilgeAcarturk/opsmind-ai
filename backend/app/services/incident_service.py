from app.repositories.incident_repository import IncidentRepository
from app.schemas.incident import Incident


class IncidentService:
    def __init__(self, repository: IncidentRepository):
        self.repository = repository

    def list_incidents(self) -> list[Incident]:
        return self.repository.get_all()

    def get_incident(self, incident_id: str) -> Incident | None:
        return self.repository.get_by_id(incident_id)
