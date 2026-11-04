from collections.abc import Iterator

from app.core.database import get_connection
from app.core.model_provider import (
    MODEL_NAME,
    get_embedding_model,
)
from app.retrieval.postgres_semantic import (
    PostgresSemanticRetriever,
)


def get_postgres_retriever() -> Iterator[PostgresSemanticRetriever]:
    model = get_embedding_model()

    with get_connection() as connection:
        yield PostgresSemanticRetriever(
            connection=connection,
            model=model,
            model_id=MODEL_NAME,
        )
