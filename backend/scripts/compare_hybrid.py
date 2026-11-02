import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from app.ingestion.document_loader import load_documents
from app.ingestion.indexer import build_index
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.keyword import KeywordRetriever
from app.retrieval.semantic import SemanticRetriever
from app.schemas.evaluation import RetrievalDataset
from app.schemas.indexing import IndexingConfig
from scripts.compare_retrievers import evaluate

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    documents = load_documents(ROOT / "data/samples/knowledge_documents_v3.json")

    dataset_path = ROOT / "evaluation/datasets/retrieval_queries_v3.json"

    dataset = RetrievalDataset.model_validate(
        json.loads(dataset_path.read_text(encoding="utf-8"))
    )

    config = IndexingConfig(chunk_size=30, overlap=5)
    chunks, manifest = build_index(documents, config)

    print("Preparing keyword retriever...")
    keyword = KeywordRetriever(chunks)

    print("Preparing BM25 retriever...")
    bm25 = BM25Retriever(chunks)

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    semantic = SemanticRetriever(chunks, model)

    print("Preparing hybrid retriever...")
    hybrid = HybridRetriever(
        bm25=bm25,
        semantic=semantic,
        candidate_k=20,
        rrf_k=60,
    )

    retrievers = {
        "keyword": keyword,
        "bm25": bm25,
        "semantic": semantic,
        "hybrid": hybrid,
    }

    results = {}

    for name, retriever in retrievers.items():
        print(f"Evaluating {name}...")
        results[name] = evaluate(
            retriever,
            dataset,
            len(chunks),
        )

    report = {
        "benchmark_version": dataset.dataset_version,
        "dataset_hash": manifest.dataset_hash,
        "evaluation_hash": stable_dataset_hash(dataset),
        "config_hash": manifest.config_hash,
        "embedding_model": MODEL_NAME,
        "hybrid_config": {
            "candidate_k": 20,
            "rrf_k": 60,
            "fusion_level": "document",
        },
        "results": results,
    }

    print("\nFOUR-WAY RETRIEVAL COMPARISON")
    print("-" * 60)

    for name, result in results.items():
        print(
            f"{name:10} | "
            f"P@2={result['precision_at_2']:.3f} | "
            f"R@2={result['recall_at_2']:.3f} | "
            f"MRR={result['mrr']:.3f}"
        )

    output_path = ROOT / "evaluation/reports/hybrid_comparison_v3.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")


def stable_dataset_hash(dataset: RetrievalDataset) -> str:
    from app.ingestion.indexer import stable_hash

    return stable_hash(dataset.model_dump(mode="json"))


if __name__ == "__main__":
    main()
