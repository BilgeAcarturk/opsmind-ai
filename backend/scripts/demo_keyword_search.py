from pathlib import Path

from app.ingestion.document_loader import load_documents
from app.ingestion.indexer import build_index
from app.retrieval.keyword import KeywordRetriever
from app.schemas.indexing import IndexingConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "samples" / "knowledge_documents.json"


def main() -> None:
    documents = load_documents(DATA_PATH)

    config = IndexingConfig(
        chunk_size=30,
        overlap=5,
    )

    chunks, manifest = build_index(documents, config)

    retriever = KeywordRetriever(chunks)

    query = "payment provider timeout errors"

    results = retriever.search(
        query=query,
        top_k=3,
        service_id="payment-service",
    )

    print(f"Dataset hash: {manifest.dataset_hash[:16]}")
    print(f"Config hash: {manifest.config_hash[:16]}")
    print(f"Query: {query}")
    print(f"Retrieved: {len(results)} results")

    for result in results:
        print("-" * 60)
        print(f"Rank: {result.rank}")
        print(f"Score: {result.score:.3f}")
        print(f"Document: {result.chunk.document_title}")
        print(f"Chunk ID: {result.chunk.id[:16]}")
        print(f"Content: {result.chunk.content}")


if __name__ == "__main__":
    main()
