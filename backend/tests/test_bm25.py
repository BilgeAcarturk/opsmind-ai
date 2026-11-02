import math

import pytest

from app.retrieval.bm25 import BM25Retriever
from app.schemas.chunk import DocumentChunk


def make_chunk(
    chunk_id: str,
    content: str,
    service_id: str = "payment-service",
) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        document_id=chunk_id,
        document_version=1,
        chunk_index=0,
        content=content,
        document_title="Test Runbook",
        document_type="runbook",
        service_id=service_id,
        source="synthetic",
    )


def test_bm25_ranking():
    chunks = [
        make_chunk("A", "payment timeout provider"),
        make_chunk("B", "database connection failure"),
        make_chunk("C", "payment authentication failure"),
    ]

    retriever = BM25Retriever(chunks)
    results = retriever.search("payment timeout")

    assert results[0].chunk.id == "A"
    assert results[0].score > 0


def test_bm25_idf():
    chunks = [
        make_chunk("A", "timeout payment"),
        make_chunk("B", "payment error"),
    ]

    retriever = BM25Retriever(chunks)

    assert retriever._idf("timeout") > retriever._idf("payment")


def test_bm25_term_frequency_saturation():
    chunks = [
        make_chunk("A", "timeout"),
        make_chunk("B", "timeout timeout"),
    ]

    retriever = BM25Retriever(chunks, b=0)

    first = retriever._score({"timeout"}, 0)
    second = retriever._score({"timeout"}, 1)

    assert second > first
    assert second < 2 * first


def test_bm25_service_filter():
    chunks = [
        make_chunk("A", "payment timeout"),
        make_chunk(
            "B",
            "payment timeout",
            service_id="order-service",
        ),
    ]

    retriever = BM25Retriever(chunks)
    results = retriever.search(
        "payment timeout",
        service_id="payment-service",
    )

    assert [r.chunk.id for r in results] == ["A"]


def test_bm25_empty_query():
    retriever = BM25Retriever([make_chunk("A", "payment timeout")])

    assert retriever.search("  ") == []


def test_bm25_no_match():
    retriever = BM25Retriever([make_chunk("A", "payment timeout")])

    assert retriever.search("database") == []


def test_bm25_invalid_parameters():
    with pytest.raises(ValueError):
        BM25Retriever([], k1=0)

    with pytest.raises(ValueError):
        BM25Retriever([], b=1.5)

    with pytest.raises(ValueError):
        BM25Retriever([]).search("payment", top_k=0)


def test_bm25_tie_breaking():
    retriever = BM25Retriever(
        [
            make_chunk("B", "payment timeout"),
            make_chunk("A", "payment timeout"),
        ]
    )

    results = retriever.search("payment")

    assert [r.chunk.id for r in results] == ["A", "B"]


def test_empty_index():
    retriever = BM25Retriever([])

    assert retriever.search("payment") == []


def test_bm25_scores_are_finite():
    retriever = BM25Retriever([make_chunk("A", "payment payment timeout")])

    results = retriever.search("payment timeout")

    assert all(math.isfinite(r.score) for r in results)
