import json
from pathlib import Path
from time import perf_counter

from sentence_transformers import SentenceTransformer

from app.evaluation.latency import summarize_latency
from app.ingestion.document_loader import load_documents
from app.ingestion.indexer import build_index, stable_hash
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.keyword import KeywordRetriever
from app.retrieval.semantic import SemanticRetriever
from app.schemas.evaluation import RetrievalDataset
from app.schemas.indexing import IndexingConfig

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
REPEATS = 10


def measure_search(
    retriever,
    dataset: RetrievalDataset,
    repeats: int = REPEATS,
) -> dict:
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    durations_ms = []

    # Warm-up: exclude first calls from latency statistics.
    for case in dataset.cases:
        retriever.search(
            case.query,
            top_k=2,
            service_id=case.service_id,
        )

    for _ in range(repeats):
        for case in dataset.cases:
            start = perf_counter()

            retriever.search(
                case.query,
                top_k=2,
                service_id=case.service_id,
            )

            elapsed_ms = (perf_counter() - start) * 1000
            durations_ms.append(elapsed_ms)

    return summarize_latency(durations_ms)


def main() -> None:
    documents = load_documents(ROOT / "data/samples/knowledge_documents_v3.json")

    raw = json.loads(
        (ROOT / "evaluation/datasets/retrieval_queries_v3.json").read_text(
            encoding="utf-8"
        )
    )
    dataset = RetrievalDataset.model_validate(raw)

    config = IndexingConfig(chunk_size=30, overlap=5)
    chunks, manifest = build_index(documents, config)

    print("Loading embedding model...")
    start = perf_counter()
    model = SentenceTransformer(MODEL_NAME)
    model_load_ms = (perf_counter() - start) * 1000

    retrievers = {}

    for name, factory in [
        ("keyword", lambda: KeywordRetriever(chunks)),
        ("bm25", lambda: BM25Retriever(chunks)),
        ("semantic", lambda: SemanticRetriever(chunks, model)),
    ]:
        start = perf_counter()
        retrievers[name] = factory()
        elapsed = (perf_counter() - start) * 1000
        print(f"{name} index initialization: {elapsed:.3f} ms")

    retrievers["hybrid"] = HybridRetriever(
        bm25=retrievers["bm25"],
        semantic=retrievers["semantic"],
        candidate_k=20,
        rrf_k=60,
        bm25_weight=0.25,
        semantic_weight=0.75,
    )

    print(f"\nModel load: {model_load_ms:.2f} ms")
    print(f"Queries: {len(dataset.cases)}")
    print(f"Repeats per query: {REPEATS}")

    results = {}

    print("\nRETRIEVAL LATENCY BENCHMARK")
    print("-" * 65)

    for name, retriever in retrievers.items():
        summary = measure_search(retriever, dataset)
        results[name] = summary

        print(
            f"{name:10} | "
            f"p50={summary['p50_ms']:.3f}ms | "
            f"p95={summary['p95_ms']:.3f}ms | "
            f"p99={summary['p99_ms']:.3f}ms"
        )

    report = {
        "benchmark_type": "local_in_process",
        "dataset_hash": manifest.dataset_hash,
        "evaluation_hash": stable_hash(dataset.model_dump(mode="json")),
        "config_hash": manifest.config_hash,
        "embedding_model": MODEL_NAME,
        "model_load_ms": model_load_ms,
        "repeats_per_query": REPEATS,
        "query_count": len(dataset.cases),
        "hybrid_weights": {
            "bm25": 0.25,
            "semantic": 0.75,
        },
        "latencies": results,
    }

    output_path = ROOT / "evaluation/reports/retrieval_latency_v3.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
