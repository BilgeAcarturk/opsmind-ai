import json
import tempfile
from pathlib import Path
from time import perf_counter

from sentence_transformers import SentenceTransformer

from app.evaluation.latency import summarize_latency
from app.ingestion.document_loader import load_documents
from app.ingestion.indexer import build_index
from app.retrieval.embedding_cache import CachedEmbeddingModel
from app.retrieval.semantic import SemanticRetriever
from app.schemas.indexing import IndexingConfig

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
REPEATS = 10


def timed_initialization(chunks, cached_model) -> float:
    start = perf_counter()
    SemanticRetriever(chunks, cached_model)
    return (perf_counter() - start) * 1000


def main() -> None:
    documents = load_documents(ROOT / "data/samples/knowledge_documents_v3.json")

    chunks, _ = build_index(
        documents,
        IndexingConfig(chunk_size=30, overlap=5),
    )

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    cold_times = []
    warm_times = []

    with tempfile.TemporaryDirectory() as directory:
        base_dir = Path(directory)

        for index in range(REPEATS):
            cache_dir = base_dir / f"trial-{index}"

            cached_model = CachedEmbeddingModel(
                model=model,
                model_id=MODEL_NAME,
                cache_dir=cache_dir,
            )

            cold_times.append(timed_initialization(chunks, cached_model))

            warm_times.append(timed_initialization(chunks, cached_model))

            assert cached_model.cache_misses == 1
            assert cached_model.cache_hits == 1

    cold = summarize_latency(cold_times)
    warm = summarize_latency(warm_times)

    print("\nEMBEDDING CACHE BENCHMARK")
    print("-" * 55)

    print(f"Cold p50: {cold['p50_ms']:.3f} ms")
    print(f"Cold p95: {cold['p95_ms']:.3f} ms")

    print(f"Warm p50: {warm['p50_ms']:.3f} ms")
    print(f"Warm p95: {warm['p95_ms']:.3f} ms")

    if warm["p50_ms"] > 0:
        speedup = cold["p50_ms"] / warm["p50_ms"]
        print(f"p50 speedup: {speedup:.1f}x")

    report = {
        "benchmark_type": "embedding_cache_initialization",
        "model": MODEL_NAME,
        "repeats": REPEATS,
        "cold": cold,
        "warm": warm,
    }

    output_path = ROOT / "evaluation/reports/embedding_cache_latency.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
