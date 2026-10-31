import pytest
from fastapi.testclient import TestClient

from app.api.incidents import get_incident_service
from app.main import app
from app.schemas.incident import Incident
from app.services.incident_service import IncidentService


class FakeIncidentRepository:
    def get_all(self) -> list[Incident]:
        return []

    def get_by_id(self, incident_id: str) -> Incident | None:
        return None


@pytest.fixture
def client():
    fake_repository = FakeIncidentRepository()
    fake_service = IncidentService(fake_repository)

    app.dependency_overrides[get_incident_service] = lambda: fake_service

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


def test_empty_incident_list(client):
    response = client.get("/api/v1/incidents")

    assert response.status_code == 200
    assert response.json() == []


def test_missing_incident_with_fake_repository(client):
    response = client.get("/api/v1/incidents/INC-001")

    assert response.status_code == 404
    assert response.json()["detail"] == "Incident not found"
