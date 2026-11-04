import hashlib

import psycopg

from app.schemas.knowledge import KnowledgeDocument


class PostgresKnowledgeRepository:
    def __init__(self, connection: psycopg.Connection):
        self._connection = connection

    def upsert_document(
        self,
        document: KnowledgeDocument,
    ) -> None:
        data = document.model_dump(mode="json")

        content_hash = hashlib.sha256(data["content"].encode("utf-8")).hexdigest()

        existing = self._connection.execute(
            """
            SELECT content_hash
            FROM knowledge_documents
            WHERE document_id = %s AND version = %s
            FOR UPDATE
            """,
            (data["id"], data["version"]),
        ).fetchone()

        if existing is not None and existing[0].strip() != content_hash:
            raise ValueError("Document content changed without a version increment")

        self._connection.execute(
            """
            INSERT INTO knowledge_documents (
                document_id,
                version,
                document_type,
                title,
                service_id,
                source,
                content,
                content_hash
            )
            VALUES (
                %(document_id)s,
                %(version)s,
                %(document_type)s,
                %(title)s,
                %(service_id)s,
                %(source)s,
                %(content)s,
                %(content_hash)s
            )
            ON CONFLICT (document_id, version)
            DO UPDATE SET
                document_type = EXCLUDED.document_type,
                title = EXCLUDED.title,
                service_id = EXCLUDED.service_id,
                source = EXCLUDED.source,
                content = EXCLUDED.content,
                content_hash = EXCLUDED.content_hash
            """,
            {
                "document_id": data["id"],
                "version": data["version"],
                "document_type": data["type"],
                "title": data["title"],
                "service_id": data["service_id"],
                "source": data["source"],
                "content": data["content"],
                "content_hash": content_hash,
            },
        )

    def count_documents(self) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) FROM knowledge_documents"
        ).fetchone()

        return row[0]
