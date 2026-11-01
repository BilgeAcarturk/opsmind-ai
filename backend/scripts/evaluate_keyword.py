import json
from pathlib import Path

from app.evaluation.metrics import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.ingestion.document_loader import load_documents
from app.ingestion.indexer import build_index
from app.retrieval.keyword import KeywordRetriever
from app.schemas.evaluation import RetrievalDataset
from app.schemas.indexing import IndexingConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def unique_document_ids(results) -> list[str]:
    seen = set()
    document_ids = []

    for result in results:
        document_id = result.chunk.document_id

        if document_id not in seen:
            seen.add(document_id)
            document_ids.append(document_id)

    return document_ids


def main() -> None:
    document_path = PROJECT_ROOT / "data" / "samples" / "knowledge_documents.json"

    evaluation_path = (
        PROJECT_ROOT / "evaluation" / "datasets" / "retrieval_queries.json"
    )

    documents = load_documents(document_path)

    raw_dataset = json.loads(evaluation_path.read_text(encoding="utf-8"))
    dataset = RetrievalDataset.model_validate(raw_dataset)

    config = IndexingConfig(chunk_size=30, overlap=5)
    chunks, manifest = build_index(documents, config)

    retriever = KeywordRetriever(chunks)

    k = 2
    results_by_query = []
    reciprocal_ranks = []

    for case in dataset.cases:
        search_results = retriever.search(
            query=case.query,
            top_k=len(chunks),
            service_id=case.service_id,
        )

        retrieved_ids = unique_document_ids(search_results)
        relevant = set(case.relevant_document_ids)

        precision = precision_at_k(retrieved_ids, relevant, k)
        recall = recall_at_k(retrieved_ids, relevant, k)
        rr = reciprocal_rank(retrieved_ids, relevant)

        reciprocal_ranks.append(rr)

        results_by_query.append(
            {
                "query_id": case.query_id,
                "query": case.query,
                "retrieved_document_ids": retrieved_ids,
                "precision_at_2": precision,
                "recall_at_2": recall,
                "reciprocal_rank": rr,
            }
        )

    count = len(results_by_query)

    summary = {
        "method": "keyword_overlap",
        "evaluation_dataset_version": dataset.dataset_version,
        "dataset_hash": manifest.dataset_hash,
        "config_hash": manifest.config_hash,
        "k": k,
        "query_count": count,
        "mean_precision_at_2": sum(r["precision_at_2"] for r in results_by_query)
        / count,
        "mean_recall_at_2": sum(r["recall_at_2"] for r in results_by_query) / count,
        "mrr": mean_reciprocal_rank(reciprocal_ranks),
        "results": results_by_query,
    }

    print(json.dumps(summary, indent=2))

    output_path = PROJECT_ROOT / "evaluation" / "reports" / "keyword_baseline.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
