from sentence_transformers import SentenceTransformer

from app.core.database import get_connection
from app.retrieval.postgres_semantic import (
    PostgresSemanticRetriever,
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    queries = [
        "payment API keys stopped working",
        "database connections are exhausted",
        "upstream gateway is returning timeout errors",
        "errors started after a new deployment",
    ]

    with get_connection() as connection:
        retriever = PostgresSemanticRetriever(
            connection=connection,
            model=model,
            model_id=MODEL_NAME,
        )

        for query in queries:
            print(f"\nQuery: {query}")
            print("-" * 55)

            results = retriever.search(
                query=query,
                top_k=3,
            )

            for result in results:
                print(
                    f"Rank {result.rank} | "
                    f"Score {result.score:.3f} | "
                    f"{result.chunk.document_id} | "
                    f"{result.chunk.document_title}"
                )


if __name__ == "__main__":
    main()
