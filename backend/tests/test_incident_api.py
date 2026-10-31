from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_list_incidents():
    response = client.get("/api/v1/incidents")

    assert response.status_code == 200

    incidents = response.json()
    assert len(incidents) == 1
    assert incidents[0]["id"] == "INC-001"


def test_get_existing_incident():
    response = client.get("/api/v1/incidents/INC-001")

    assert response.status_code == 200
    assert response.json()["severity"] == "SEV2"


def test_get_missing_incident():
    response = client.get("/api/v1/incidents/INC-999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Incident not found"
