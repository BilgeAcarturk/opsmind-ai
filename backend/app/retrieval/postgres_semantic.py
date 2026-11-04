from typing import Protocol

import numpy as np
import psycopg
from numpy.typing import NDArray
from pgvector import Vector

from app.schemas.chunk import DocumentChunk
from app.schemas.retrieval import SearchResult


class QueryEncoder(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> NDArray[np.float32]: ...


class PostgresSemanticRetriever:
    def __init__(
        self,
        connection: psycopg.Connection,
        model: QueryEncoder,
        model_id: str,
    ):
        if not model_id.strip():
            raise ValueError("model_id cannot be empty")

        self._connection = connection
        self._model = model
        self._model_id = model_id

    def search(
        self,
        query: str,
        top_k: int = 5,
        service_id: str | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if not query.strip():
            return []

        embedding = np.asarray(
            self._model.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )

        if embedding.shape != (1, 384):
            raise ValueError("Invalid query embedding shape")

        if not np.isfinite(embedding).all():
            raise ValueError("Invalid query embedding values")

        query_vector = Vector(embedding[0])

        rows = self._connection.execute(
            """
            SELECT
                c.chunk_id,
                c.document_id,
                c.document_version,
                c.chunk_index,
                c.content,
                d.title,
                d.document_type,
                d.service_id,
                d.source,
                1 - (e.embedding <=> %s) AS similarity
            FROM chunk_embeddings AS e
            JOIN document_chunks AS c
                ON e.chunk_id = c.chunk_id
            JOIN knowledge_documents AS d
                ON c.document_id = d.document_id
                AND c.document_version = d.version
            WHERE e.model_id = %s
              AND (%s::text IS NULL OR d.service_id = %s)
            ORDER BY e.embedding <=> %s, c.chunk_id
            LIMIT %s
            """,
            (
                query_vector,
                self._model_id,
                service_id,
                service_id,
                query_vector,
                top_k,
            ),
        ).fetchall()

        results = []

        for rank, row in enumerate(rows, start=1):
            chunk = DocumentChunk(
                id=row[0].strip(),
                document_id=row[1],
                document_version=row[2],
                chunk_index=row[3],
                content=row[4],
                document_title=row[5],
                document_type=row[6],
                service_id=row[7],
                source=row[8],
            )

            similarity = float(row[9])

            results.append(
                SearchResult(
                    chunk=chunk,
                    score=max(-1.0, min(1.0, similarity)),
                    rank=rank,
                )
            )

        return results
