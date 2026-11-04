import psycopg
import pytest
from fastapi.testclient import TestClient

from app.api.retrieval_dependencies import (
    get_postgres_retriever,
)
from app.main import app


class FailingRetriever:
    def search(self, query, top_k=5, service_id=None):
        raise psycopg.OperationalError("Simulated database connection failure")


class TrackingRetriever:
    def __init__(self):
        self.calls = 0

    def search(self, query, top_k=5, service_id=None):
        self.calls += 1
        return []


@pytest.fixture
def client_with_failure():
    app.dependency_overrides[get_postgres_retriever] = lambda: FailingRetriever()

    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_postgres_retriever, None)


def test_database_failure_returns_503(client_with_failure):
    response = client_with_failure.post(
        "/api/v1/retrieval/search",
        json={
            "query": "payment provider timeout",
            "top_k": 3,
        },
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Retrieval service temporarily unavailable"}


def test_database_error_details_are_not_exposed(
    client_with_failure,
):
    response = client_with_failure.post(
        "/api/v1/retrieval/search",
        json={"query": "payment timeout"},
    )

    assert response.status_code == 503
    assert "Simulated database" not in response.text


def test_invalid_input_does_not_search():
    retriever = TrackingRetriever()

    app.dependency_overrides[get_postgres_retriever] = lambda: retriever

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/retrieval/search",
                json={
                    "query": "payment timeout",
                    "top_k": -1,
                },
            )

        assert response.status_code == 422
        assert retriever.calls == 0

    finally:
        app.dependency_overrides.pop(get_postgres_retriever, None)


from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi import HTTPException
from psycopg_pool import PoolTimeout

from app.api import retrieval_dependencies


def test_connection_pool_timeout_returns_503(monkeypatch):
    fake_pool = MagicMock()
    fake_pool.connection.side_effect = PoolTimeout("No connection available")

    monkeypatch.setattr(
        retrieval_dependencies,
        "get_embedding_model",
        lambda: object(),
    )

    monkeypatch.setattr(
        retrieval_dependencies,
        "get_or_create_pool",
        lambda request: fake_pool,
    )

    fake_request = SimpleNamespace()

    dependency = retrieval_dependencies.get_postgres_retriever(fake_request)

    with pytest.raises(HTTPException) as exc_info:
        next(dependency)

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == ("Database temporarily unavailable")
