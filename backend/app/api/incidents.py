from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.repositories.incident_repository import (
    InMemoryIncidentRepository,
)
from app.schemas.incident import Incident
from app.services.incident_service import IncidentService

router = APIRouter(prefix="/api/v1/incidents", tags=["Incidents"])

_repository = InMemoryIncidentRepository()


def get_incident_service() -> IncidentService:
    return IncidentService(repository=_repository)


@router.get("", response_model=list[Incident])
def list_incidents(
    service: Annotated[IncidentService, Depends(get_incident_service)],
):
    return service.list_incidents()


@router.get("/{incident_id}", response_model=Incident)
def get_incident(
    incident_id: str,
    service: Annotated[IncidentService, Depends(get_incident_service)],
):
    incident = service.get_incident(incident_id)

    if incident is None:
        raise HTTPException(
            status_code=404,
            detail="Incident not found",
        )

    return incident
