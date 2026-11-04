import pytest
from fastapi.testclient import TestClient

from app.api.retrieval_dependencies import get_postgres_retriever
from app.main import app
from app.schemas.chunk import DocumentChunk
from app.schemas.retrieval import SearchResult


class FakeRetriever:
    def search(
        self,
        query: str,
        top_k: int = 5,
        service_id: str | None = None,
    ) -> list[SearchResult]:
        chunk = DocumentChunk(
            id="test-chunk-001",
            document_id="RB-003",
            document_version=1,
            chunk_index=0,
            content="Payment credentials expired.",
            document_title="Payment Authentication Failure",
            document_type="runbook",
            service_id="payment-service",
            source="synthetic",
        )

        if service_id not in (None, "payment-service"):
            return []

        return [
            SearchResult(
                chunk=chunk,
                score=0.85,
                rank=1,
            )
        ][:top_k]


@pytest.fixture
def client():
    app.dependency_overrides[get_postgres_retriever] = lambda: FakeRetriever()

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(get_postgres_retriever, None)


def test_retrieval_search_success(client):
    response = client.post(
        "/api/v1/retrieval/search",
        json={
            "query": "payment API keys stopped working",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["method"] == "postgres_semantic"
    assert data["total_results"] == 1
    assert data["results"][0]["chunk"]["document_id"] == "RB-003"


def test_retrieval_short_query_rejected(client):
    response = client.post(
        "/api/v1/retrieval/search",
        json={"query": "x"},
    )

    assert response.status_code == 422


def test_retrieval_whitespace_query_rejected(client):
    response = client.post(
        "/api/v1/retrieval/search",
        json={"query": "     "},
    )

    assert response.status_code == 422


def test_retrieval_invalid_top_k(client):
    response = client.post(
        "/api/v1/retrieval/search",
        json={"query": "payment timeout", "top_k": 0},
    )

    assert response.status_code == 422


def test_retrieval_service_filter(client):
    response = client.post(
        "/api/v1/retrieval/search",
        json={
            "query": "payment timeout",
            "service_id": "unknown-service",
        },
    )

    assert response.status_code == 200
    assert response.json()["total_results"] == 0


def test_retrieval_extra_fields_rejected(client):
    response = client.post(
        "/api/v1/retrieval/search",
        json={
            "query": "payment timeout",
            "unexpected": "value",
        },
    )

    assert response.status_code == 422
