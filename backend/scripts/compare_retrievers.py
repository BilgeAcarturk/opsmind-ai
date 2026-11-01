import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from app.evaluation.metrics import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.ingestion.document_loader import load_documents
from app.ingestion.indexer import build_index
from app.retrieval.keyword import KeywordRetriever
from app.retrieval.semantic import SemanticRetriever
from app.schemas.evaluation import RetrievalDataset
from app.schemas.indexing import IndexingConfig

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def unique_document_ids(results) -> list[str]:
    seen = set()
    unique_ids = []

    for result in results:
        document_id = result.chunk.document_id

        if document_id not in seen:
            seen.add(document_id)
            unique_ids.append(document_id)

    return unique_ids


def evaluate(retriever, dataset, chunk_count, k=2):
    case_results = []

    for case in dataset.cases:
        results = retriever.search(
            query=case.query,
            top_k=chunk_count,
            service_id=case.service_id,
        )

        retrieved = unique_document_ids(results)
        relevant = set(case.relevant_document_ids)

        case_results.append(
            {
                "query_id": case.query_id,
                "retrieved_ids": retrieved,
                "precision_at_k": precision_at_k(retrieved, relevant, k),
                "recall_at_k": recall_at_k(retrieved, relevant, k),
                "rr": reciprocal_rank(retrieved, relevant),
            }
        )

    count = len(case_results)

    return {
        "precision_at_2": sum(r["precision_at_k"] for r in case_results) / count,
        "recall_at_2": sum(r["recall_at_k"] for r in case_results) / count,
        "mrr": mean_reciprocal_rank([r["rr"] for r in case_results]),
        "cases": case_results,
    }


def main():
    documents = load_documents(ROOT / "data/samples/knowledge_documents.json")

    raw_dataset = json.loads(
        (ROOT / "evaluation/datasets/retrieval_queries.json").read_text(
            encoding="utf-8"
        )
    )

    dataset = RetrievalDataset.model_validate(raw_dataset)

    config = IndexingConfig(chunk_size=30, overlap=5)
    chunks, manifest = build_index(documents, config)

    print("Evaluating keyword retriever...")
    keyword = KeywordRetriever(chunks)

    keyword_results = evaluate(keyword, dataset, len(chunks))

    print("Loading semantic embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    semantic = SemanticRetriever(chunks, model)

    print("Evaluating semantic retriever...")
    semantic_results = evaluate(semantic, dataset, len(chunks))

    comparison = {
        "dataset_hash": manifest.dataset_hash,
        "config_hash": manifest.config_hash,
        "evaluation_version": dataset.dataset_version,
        "model": MODEL_NAME,
        "keyword": keyword_results,
        "semantic": semantic_results,
    }

    print("\nRETRIEVAL COMPARISON")
    print("-" * 45)

    for metric in ["precision_at_2", "recall_at_2", "mrr"]:
        keyword_value = keyword_results[metric]
        semantic_value = semantic_results[metric]

        print(f"{metric}: keyword={keyword_value:.3f} | semantic={semantic_value:.3f}")

    output_path = ROOT / "evaluation/reports/retrieval_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(comparison, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
