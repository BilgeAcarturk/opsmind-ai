import numpy as np
import pytest

from app.retrieval.postgres_semantic import (
    PostgresSemanticRetriever,
)


class FakeModel:
    def encode(
        self,
        sentences,
        *,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ):
        return np.ones((len(sentences), 384), dtype=np.float32)


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, rows=None):
        self.rows = rows if rows is not None else []
        self.calls = []

    def execute(self, query, params):
        self.calls.append((query, params))
        return FakeCursor(self.rows)


def make_row():
    return (
        "a" * 64,
        "RB-003",
        1,
        0,
        "Payment credentials expired.",
        "Payment Authentication Failure",
        "runbook",
        "payment-service",
        "synthetic",
        0.85,
    )


def test_postgres_semantic_ranking():
    connection = FakeConnection([make_row()])

    retriever = PostgresSemanticRetriever(
        connection,
        FakeModel(),
        "test-model",
    )

    results = retriever.search("payment credentials", top_k=3)

    assert len(results) == 1
    assert results[0].chunk.document_id == "RB-003"
    assert results[0].rank == 1
    assert results[0].score == pytest.approx(0.85)


def test_empty_query_does_not_access_database():
    connection = FakeConnection()

    retriever = PostgresSemanticRetriever(
        connection,
        FakeModel(),
        "test-model",
    )

    assert retriever.search("   ") == []
    assert connection.calls == []


def test_invalid_top_k():
    retriever = PostgresSemanticRetriever(
        FakeConnection(),
        FakeModel(),
        "test-model",
    )

    with pytest.raises(ValueError):
        retriever.search("payment", top_k=0)


def test_invalid_model_id():
    with pytest.raises(ValueError):
        PostgresSemanticRetriever(
            FakeConnection(),
            FakeModel(),
            "",
        )


def test_query_uses_model_and_service_filter():
    connection = FakeConnection()

    retriever = PostgresSemanticRetriever(
        connection,
        FakeModel(),
        "test-model",
    )

    retriever.search(
        "payment timeout",
        top_k=2,
        service_id="payment-service",
    )

    assert len(connection.calls) == 1

    sql, params = connection.calls[0]

    assert "e.model_id = %s" in sql
    assert params[1] == "test-model"
    assert params[2] == "payment-service"
    assert params[3] == "payment-service"
    assert params[5] == 2


def test_no_results():
    retriever = PostgresSemanticRetriever(
        FakeConnection([]),
        FakeModel(),
        "test-model",
    )

    assert retriever.search("unknown incident") == []
