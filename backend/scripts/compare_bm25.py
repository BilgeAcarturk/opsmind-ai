import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from app.ingestion.document_loader import load_documents
from app.ingestion.indexer import build_index
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.keyword import KeywordRetriever
from app.retrieval.semantic import SemanticRetriever
from app.schemas.evaluation import RetrievalDataset
from app.schemas.indexing import IndexingConfig
from scripts.compare_retrievers import evaluate

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    documents = load_documents(ROOT / "data/samples/knowledge_documents_v3.json")

    raw_dataset = json.loads(
        (ROOT / "evaluation/datasets/retrieval_queries_v3.json").read_text(
            encoding="utf-8"
        )
    )

    dataset = RetrievalDataset.model_validate(raw_dataset)

    config = IndexingConfig(chunk_size=30, overlap=5)
    chunks, manifest = build_index(documents, config)

    print("Running keyword evaluation...")
    keyword = KeywordRetriever(chunks)
    keyword_results = evaluate(keyword, dataset, len(chunks))

    print("Running BM25 evaluation...")
    bm25 = BM25Retriever(chunks)
    bm25_results = evaluate(bm25, dataset, len(chunks))

    print("Running semantic evaluation...")
    model = SentenceTransformer(MODEL_NAME)
    semantic = SemanticRetriever(chunks, model)
    semantic_results = evaluate(semantic, dataset, len(chunks))

    results = {
        "benchmark_version": dataset.dataset_version,
        "dataset_hash": manifest.dataset_hash,
        "config_hash": manifest.config_hash,
        "model": MODEL_NAME,
        "bm25_parameters": {"k1": 1.5, "b": 0.75},
        "keyword": keyword_results,
        "bm25": bm25_results,
        "semantic": semantic_results,
    }

    print("\nTHREE-WAY RETRIEVAL COMPARISON")
    print("-" * 60)

    for metric in ["precision_at_2", "recall_at_2", "mrr"]:
        print(
            f"{metric}: "
            f"keyword={keyword_results[metric]:.3f} | "
            f"bm25={bm25_results[metric]:.3f} | "
            f"semantic={semantic_results[metric]:.3f}"
        )

    output_path = ROOT / "evaluation/reports/bm25_comparison_v3.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
