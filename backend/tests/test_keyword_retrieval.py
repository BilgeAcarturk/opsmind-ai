import pytest

from app.retrieval.keyword import (
    KeywordRetriever,
    keyword_score,
    tokenize,
)
from app.schemas.chunk import DocumentChunk


def make_chunk(
    chunk_id: str,
    content: str,
    service_id: str = "payment-service",
) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        document_id="RB-TEST",
        document_version=1,
        chunk_index=0,
        content=content,
        document_title="Test Runbook",
        document_type="runbook",
        service_id=service_id,
        source="synthetic",
    )


def test_tokenization():
    assert tokenize("Payment, TIMEOUT!") == {
        "payment",
        "timeout",
    }


def test_keyword_score():
    score = keyword_score(
        "payment provider timeout",
        "Payment provider latency and timeout",
    )

    assert score == 1.0


def test_no_matching_terms():
    score = keyword_score(
        "database connection",
        "Payment provider timeout",
    )

    assert score == 0.0


def test_retrieval_ranking():
    chunks = [
        make_chunk("A", "payment failure"),
        make_chunk("B", "payment provider timeout"),
        make_chunk("C", "database latency"),
    ]

    retriever = KeywordRetriever(chunks)

    results = retriever.search(
        "payment provider timeout",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk.id == "B"
    assert results[0].score == 1.0
    assert results[0].rank == 1
    assert results[1].chunk.id == "A"


def test_service_filter():
    chunks = [
        make_chunk(
            "A",
            "payment timeout",
            service_id="payment-service",
        ),
        make_chunk(
            "B",
            "payment timeout",
            service_id="order-service",
        ),
    ]

    retriever = KeywordRetriever(chunks)

    results = retriever.search(
        "payment timeout",
        service_id="payment-service",
    )

    assert len(results) == 1
    assert results[0].chunk.id == "A"


def test_no_results():
    retriever = KeywordRetriever([make_chunk("A", "payment timeout")])

    assert retriever.search("database connection") == []


def test_invalid_top_k():
    retriever = KeywordRetriever([])

    with pytest.raises(ValueError):
        retriever.search("payment", top_k=0)


def test_deterministic_tie_breaking():
    chunks = [
        make_chunk("B", "payment timeout"),
        make_chunk("A", "payment timeout"),
    ]

    retriever = KeywordRetriever(chunks)
    results = retriever.search("payment")

    assert [result.chunk.id for result in results] == [
        "A",
        "B",
    ]


def test_empty_query():
    retriever = KeywordRetriever([make_chunk("A", "payment timeout")])

    assert retriever.search("   ") == []
