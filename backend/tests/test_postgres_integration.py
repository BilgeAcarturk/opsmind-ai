import os

import pytest

from app.core.database import get_connection

pytestmark = pytest.mark.skipif(
    os.getenv("OPSMIND_RUN_DB_TESTS") != "1",
    reason="PostgreSQL integration tests are opt-in",
)


def test_postgres_connection_and_pgvector():
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                current_database(),
                (
                    SELECT extversion
                    FROM pg_extension
                    WHERE extname = 'vector'
                )
            """
        ).fetchone()

    assert row[0] == "opsmind"
    assert row[1] == "0.8.7"


def test_postgres_tables_exist():
    expected_tables = {
        "knowledge_documents",
        "document_chunks",
        "chunk_embeddings",
    }

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_type = 'BASE TABLE'
            """
        ).fetchall()

    actual_tables = {row[0] for row in rows}

    assert expected_tables <= actual_tables


def test_persisted_embeddings_have_correct_dimensions():
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                COUNT(*),
                COUNT(*) FILTER (
                    WHERE vector_dims(embedding) = 384
                )
            FROM chunk_embeddings
            """
        ).fetchone()

    total, valid = row

    assert total > 0
    assert total == valid


def test_postgres_nullable_service_filter():
    import numpy as np

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
            vectors = np.zeros(
                (len(sentences), 384),
                dtype=np.float32,
            )
            vectors[:, 0] = 1.0
            return vectors

    with get_connection() as connection:
        retriever = PostgresSemanticRetriever(
            connection=connection,
            model=FakeModel(),
            model_id="sentence-transformers/all-MiniLM-L6-v2",
        )

        unfiltered = retriever.search(
            "payment timeout",
            top_k=3,
            service_id=None,
        )

        filtered = retriever.search(
            "payment timeout",
            top_k=3,
            service_id="payment-service",
        )

    assert unfiltered
    assert filtered
    assert all(item.chunk.service_id == "payment-service" for item in filtered)
