from pathlib import Path

from sentence_transformers import SentenceTransformer

from app.core.database import get_connection
from app.ingestion.document_loader import load_documents
from app.ingestion.indexer import build_index
from app.retrieval.postgres_semantic import (
    PostgresSemanticRetriever,
)
from app.retrieval.semantic import SemanticRetriever
from app.schemas.indexing import IndexingConfig

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

QUERIES = [
    "payment API keys stopped working",
    "database connections are exhausted",
    "upstream gateway is returning timeout errors",
    "errors started after a new deployment",
]

SCORE_TOLERANCE = 0.001


def compare_results(
    numpy_results,
    postgres_results,
) -> tuple[bool, float]:
    if len(numpy_results) != len(postgres_results):
        return False, float("inf")

    numpy_ids = [result.chunk.id for result in numpy_results]
    postgres_ids = [result.chunk.id for result in postgres_results]

    ids_match = numpy_ids == postgres_ids

    differences = [
        abs(n.score - p.score)
        for n, p in zip(
            numpy_results,
            postgres_results,
            strict=True,
        )
    ]

    max_difference = max(differences, default=0.0)

    return (
        ids_match and max_difference <= SCORE_TOLERANCE,
        max_difference,
    )


def main() -> None:
    documents = load_documents(ROOT / "data/samples/knowledge_documents_v3.json")

    chunks, _ = build_index(
        documents,
        IndexingConfig(chunk_size=30, overlap=5),
    )

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    numpy_retriever = SemanticRetriever(chunks, model)

    all_passed = True

    with get_connection() as connection:
        postgres_retriever = PostgresSemanticRetriever(
            connection=connection,
            model=model,
            model_id=MODEL_NAME,
        )

        for query in QUERIES:
            numpy_results = numpy_retriever.search(query, top_k=5)

            postgres_results = postgres_retriever.search(query, top_k=5)

            passed, max_difference = compare_results(
                numpy_results,
                postgres_results,
            )

            all_passed = all_passed and passed

            print(f"\nQuery: {query}")
            print(f"Parity: {'PASS' if passed else 'FAIL'}")
            print(f"Max score difference: {max_difference:.6f}")

            for rank, (n, p) in enumerate(
                zip(
                    numpy_results,
                    postgres_results,
                    strict=False,
                ),
                start=1,
            ):
                print(
                    f"Rank {rank}: "
                    f"NumPy={n.chunk.id[:12]} "
                    f"({n.score:.4f}) | "
                    f"Postgres={p.chunk.id[:12]} "
                    f"({p.score:.4f})"
                )

    print("\nPARITY SUMMARY")
    print("PASS" if all_passed else "FAIL")

    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
