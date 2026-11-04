from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.main import app


def test_liveness():
    with TestClient(app) as client:
        response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_readiness_without_pool():
    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503


def test_readiness_with_healthy_resources(monkeypatch):
    fake_pool = MagicMock()
    fake_pool.closed = False

    connection = MagicMock()
    connection.execute.return_value.fetchone.return_value = (1,)

    fake_pool.connection.return_value.__enter__.return_value = connection

    monkeypatch.setattr(
        "app.api.health.get_embedding_model",
        SimpleNamespace(cache_info=lambda: SimpleNamespace(currsize=1)),
    )

    with TestClient(app) as client:
        app.state.db_pool = fake_pool

        response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    fake_pool.connection.assert_called_once()
