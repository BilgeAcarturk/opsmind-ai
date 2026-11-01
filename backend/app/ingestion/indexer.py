import hashlib
import json
from datetime import UTC, datetime
from uuid import uuid4

from app.ingestion.chunker import chunk_document
from app.schemas.chunk import DocumentChunk
from app.schemas.indexing import IndexingConfig, IngestionManifest
from app.schemas.knowledge import KnowledgeDocument


def stable_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_index(
    documents: list[KnowledgeDocument],
    config: IndexingConfig,
) -> tuple[list[DocumentChunk], IngestionManifest]:
    document_ids = [document.id for document in documents]

    if len(document_ids) != len(set(document_ids)):
        raise ValueError("Duplicate document IDs detected")

    config_hash = stable_hash(config.model_dump())

    chunks: list[DocumentChunk] = []
    document_hashes: dict[str, str] = {}

    for document in sorted(documents, key=lambda item: item.id):
        document_hash = stable_hash(document.model_dump(mode="json"))
        document_hashes[document.id] = document_hash

        document_chunks = chunk_document(
            document,
            chunk_size=config.chunk_size,
            overlap=config.overlap,
        )

        for chunk in document_chunks:
            chunk_id = stable_hash(
                {
                    "document_hash": document_hash,
                    "config_hash": config_hash,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                }
            )

            chunks.append(chunk.model_copy(update={"id": chunk_id}))

    dataset_hash = stable_hash(document_hashes)

    manifest = IngestionManifest(
        run_id=str(uuid4()),
        created_at=datetime.now(UTC),
        config=config,
        config_hash=config_hash,
        dataset_hash=dataset_hash,
        document_count=len(documents),
        chunk_count=len(chunks),
        document_hashes=document_hashes,
    )

    return chunks, manifest
