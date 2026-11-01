from pathlib import Path

from sentence_transformers import SentenceTransformer

from app.ingestion.document_loader import load_documents
from app.ingestion.indexer import build_index
from app.retrieval.semantic import SemanticRetriever
from app.schemas.indexing import IndexingConfig

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    documents = load_documents(ROOT / "data/samples/knowledge_documents.json")

    chunks, manifest = build_index(
        documents,
        IndexingConfig(chunk_size=30, overlap=5),
    )

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    retriever = SemanticRetriever(chunks, model)

    queries = [
        "payment provider timeout errors",
        "external payment gateway stopped responding",
        "database connections are exhausted",
    ]

    print("Dataset hash:", manifest.dataset_hash)
    print("Config hash:", manifest.config_hash)

    for query in queries:
        print(f"\nQuery: {query}")

        results = retriever.search(query, top_k=3)

        for result in results:
            print(
                f"Rank {result.rank} | "
                f"Score {result.score:.3f} | "
                f"{result.chunk.document_title}"
            )


if __name__ == "__main__":
    main()
