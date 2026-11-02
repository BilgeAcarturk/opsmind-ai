import json
from pathlib import Path

from app.ingestion.document_loader import load_documents
from app.schemas.evaluation import RetrievalDataset

ROOT = Path(__file__).resolve().parents[2]


def load_benchmark():
    documents = load_documents(ROOT / "data/samples/knowledge_documents_v3.json")

    raw = json.loads(
        (ROOT / "evaluation/datasets/retrieval_queries_v3.json").read_text(
            encoding="utf-8"
        )
    )

    dataset = RetrievalDataset.model_validate(raw)
    return documents, dataset


def test_benchmark_document_and_query_counts():
    documents, dataset = load_benchmark()

    assert len(documents) == 9
    assert len(dataset.cases) == 8


def test_relevance_labels_reference_existing_documents():
    documents, dataset = load_benchmark()

    known_ids = {document.id for document in documents}

    for case in dataset.cases:
        assert set(case.relevant_document_ids) <= known_ids


def test_relevant_documents_match_service_filters():
    documents, dataset = load_benchmark()

    by_id = {document.id: document for document in documents}

    for case in dataset.cases:
        if case.service_id is None:
            continue

        for document_id in case.relevant_document_ids:
            assert by_id[document_id].service_id == case.service_id


def test_query_ids_are_unique():
    _, dataset = load_benchmark()

    query_ids = [case.query_id for case in dataset.cases]

    assert len(query_ids) == len(set(query_ids))
