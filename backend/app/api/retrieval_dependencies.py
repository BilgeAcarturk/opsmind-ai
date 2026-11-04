from collections.abc import Iterator

import psycopg
from fastapi import HTTPException, Request
from psycopg_pool import PoolClosed, PoolTimeout

from app.core.database_pool import get_or_create_pool
from app.core.model_provider import (
    MODEL_NAME,
    get_embedding_model,
)
from app.retrieval.postgres_semantic import (
    PostgresSemanticRetriever,
)


def get_postgres_retriever(
    request: Request,
) -> Iterator[PostgresSemanticRetriever]:
    model = get_embedding_model()

    try:
        pool = get_or_create_pool(request)

        with pool.connection(timeout=5.0) as connection:
            yield PostgresSemanticRetriever(
                connection=connection,
                model=model,
                model_id=MODEL_NAME,
            )

    except (psycopg.Error, PoolTimeout, PoolClosed) as exc:
        raise HTTPException(
            status_code=503,
            detail="Database temporarily unavailable",
        ) from exc
