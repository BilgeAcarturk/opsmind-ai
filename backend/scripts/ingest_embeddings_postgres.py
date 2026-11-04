from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.database import get_connection
from app.ingestion.document_loader import load_documents
from app.ingestion.indexer import build_index
from app.repositories.knowledge_repository import (
    PostgresKnowledgeRepository,
)
from app.repositories.vector_repository import (
    PostgresVectorRepository,
)
from app.schemas.indexing import IndexingConfig

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    documents = load_documents(ROOT / "data/samples/knowledge_documents_v3.json")

    config = IndexingConfig(
        chunk_size=30,
        overlap=5,
    )

    chunks, manifest = build_index(documents, config)

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks generated: {len(chunks)}")

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    embeddings = np.asarray(
        model.encode(
            [chunk.content for chunk in chunks],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )

    if embeddings.shape != (len(chunks), 384):
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")

    if not np.isfinite(embeddings).all():
        raise ValueError("Invalid embedding values")

    print(f"Embedding matrix: {embeddings.shape}")

    # All database writes belong to one transaction.
    with get_connection() as connection:
        knowledge_repo = PostgresKnowledgeRepository(connection)
        vector_repo = PostgresVectorRepository(connection)

        for document in documents:
            knowledge_repo.upsert_document(document)

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            vector_repo.upsert_chunk(
                chunk,
                config_hash=manifest.config_hash,
            )

            vector_repo.upsert_embedding(
                chunk_id=chunk.id,
                model_id=MODEL_NAME,
                embedding=embedding,
            )

        document_count = knowledge_repo.count_documents()
        chunk_count = vector_repo.count_chunks()
        embedding_count = vector_repo.count_embeddings(MODEL_NAME)

    print("\nDATABASE INGESTION COMPLETE")
    print("-" * 45)
    print(f"Documents in database: {document_count}")
    print(f"Chunks in database: {chunk_count}")
    print(f"Embeddings for model: {embedding_count}")
    print(f"Dataset hash: {manifest.dataset_hash[:16]}")
    print(f"Config hash: {manifest.config_hash[:16]}")


if __name__ == "__main__":
    main()
