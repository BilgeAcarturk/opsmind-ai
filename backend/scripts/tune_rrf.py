import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from app.ingestion.document_loader import load_documents
from app.ingestion.indexer import build_index, stable_hash
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.semantic import SemanticRetriever
from app.schemas.evaluation import RetrievalDataset
from app.schemas.indexing import IndexingConfig
from scripts.compare_retrievers import evaluate

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    documents = load_documents(ROOT / "data/samples/knowledge_documents_v3.json")

    dataset = RetrievalDataset.model_validate(
        json.loads(
            (ROOT / "evaluation/datasets/retrieval_queries_tuning.json").read_text(
                encoding="utf-8"
            )
        )
    )

    config = IndexingConfig(chunk_size=30, overlap=5)
    chunks, manifest = build_index(documents, config)

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    bm25 = BM25Retriever(chunks)
    semantic = SemanticRetriever(chunks, model)

    weights = [0.0, 0.25, 0.5, 0.75, 1.0]

    experiments = []

    print("\nWEIGHTED RRF TUNING")
    print("-" * 65)

    for semantic_weight in weights:
        bm25_weight = 1.0 - semantic_weight

        hybrid = HybridRetriever(
            bm25=bm25,
            semantic=semantic,
            candidate_k=20,
            rrf_k=60,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
        )

        result = evaluate(
            hybrid,
            dataset,
            len(chunks),
        )

        experiments.append(
            {
                "bm25_weight": bm25_weight,
                "semantic_weight": semantic_weight,
                "metrics": result,
            }
        )

        print(
            f"BM25={bm25_weight:.2f} | "
            f"Semantic={semantic_weight:.2f} | "
            f"P@2={result['precision_at_2']:.3f} | "
            f"R@2={result['recall_at_2']:.3f} | "
            f"MRR={result['mrr']:.3f}"
        )

    report = {
        "purpose": "development_tuning",
        "dataset_hash": manifest.dataset_hash,
        "config_hash": manifest.config_hash,
        "evaluation_hash": stable_hash(dataset.model_dump(mode="json")),
        "embedding_model": MODEL_NAME,
        "candidate_k": 20,
        "rrf_k": 60,
        "experiments": experiments,
    }

    output_path = ROOT / "evaluation/reports/rrf_weight_tuning.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
