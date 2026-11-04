import numpy as np
import psycopg
from pgvector import Vector

from app.schemas.chunk import DocumentChunk


class PostgresVectorRepository:
    def __init__(self, connection: psycopg.Connection):
        self._connection = connection

    def upsert_chunk(
        self,
        chunk: DocumentChunk,
        config_hash: str,
    ) -> None:
        existing = self._connection.execute(
            """
            SELECT content, config_hash
            FROM document_chunks
            WHERE chunk_id = %s
            FOR UPDATE
            """,
            (chunk.id,),
        ).fetchone()

        if existing is not None and (
            existing[0] != chunk.content or existing[1].strip() != config_hash
        ):
            raise ValueError("Chunk ID conflicts with existing content")

        self._connection.execute(
            """
            INSERT INTO document_chunks (
                chunk_id,
                document_id,
                document_version,
                chunk_index,
                config_hash,
                content
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO NOTHING
            """,
            (
                chunk.id,
                chunk.document_id,
                chunk.document_version,
                chunk.chunk_index,
                config_hash,
                chunk.content,
            ),
        )

    def upsert_embedding(
        self,
        chunk_id: str,
        model_id: str,
        embedding: np.ndarray,
    ) -> None:
        vector = np.asarray(embedding, dtype=np.float32)

        if vector.shape != (384,):
            raise ValueError(f"Expected 384 dimensions, got {vector.shape}")

        if not np.isfinite(vector).all():
            raise ValueError("Embedding contains invalid values")

        if not model_id.strip():
            raise ValueError("model_id cannot be empty")

        self._connection.execute(
            """
            INSERT INTO chunk_embeddings (
                chunk_id,
                model_id,
                embedding
            )
            VALUES (%s, %s, %s)
            ON CONFLICT (chunk_id, model_id)
            DO UPDATE SET
                embedding = EXCLUDED.embedding
            """,
            (
                chunk_id,
                model_id,
                Vector(vector),
            ),
        )

    def count_chunks(self) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) FROM document_chunks"
        ).fetchone()
        return row[0]

    def count_embeddings(self, model_id: str) -> int:
        row = self._connection.execute(
            """
            SELECT COUNT(*)
            FROM chunk_embeddings
            WHERE model_id = %s
            """,
            (model_id,),
        ).fetchone()
        return row[0]
