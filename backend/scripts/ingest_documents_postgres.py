from pathlib import Path

from app.core.database import get_connection
from app.ingestion.document_loader import load_documents
from app.repositories.knowledge_repository import (
    PostgresKnowledgeRepository,
)

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    documents = load_documents(ROOT / "data/samples/knowledge_documents_v3.json")

    with get_connection() as connection:
        repository = PostgresKnowledgeRepository(connection)

        with connection.transaction():
            for document in documents:
                repository.upsert_document(document)

            total = repository.count_documents()

    print(f"Documents processed: {len(documents)}")
    print(f"Documents in database: {total}")


if __name__ == "__main__":
    main()
