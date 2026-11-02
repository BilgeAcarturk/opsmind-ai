from pathlib import Path
from time import perf_counter

from sentence_transformers import SentenceTransformer

from app.ingestion.document_loader import load_documents
from app.ingestion.indexer import build_index
from app.retrieval.embedding_cache import CachedEmbeddingModel
from app.retrieval.semantic import SemanticRetriever
from app.schemas.indexing import IndexingConfig

ROOT = Path(__file__).resolve().parents[2]

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CACHE_DIR = ROOT / "backend" / ".cache" / "embeddings"


def main() -> None:
    documents = load_documents(ROOT / "data/samples/knowledge_documents_v3.json")

    chunks, manifest = build_index(
        documents,
        IndexingConfig(chunk_size=30, overlap=5),
    )

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    cached_model = CachedEmbeddingModel(
        model=model,
        model_id=MODEL_NAME,
        cache_dir=CACHE_DIR,
    )

    start = perf_counter()

    retriever = SemanticRetriever(
        chunks,
        cached_model,
    )

    elapsed = perf_counter() - start

    print(f"Dataset: {manifest.dataset_hash[:16]}")
    print(f"Index initialization: {elapsed:.4f}s")
    print(f"Cache hits: {cached_model.cache_hits}")
    print(f"Cache misses: {cached_model.cache_misses}")

    results = retriever.search(
        "payment gateway authentication failure",
        top_k=3,
    )

    for result in results:
        print(f"{result.rank}. {result.chunk.document_title} ({result.score:.3f})")


if __name__ == "__main__":
    main()
