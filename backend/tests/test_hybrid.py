import pytest

from app.retrieval.hybrid import (
    HybridRetriever,
    reciprocal_rank_fusion,
)
from app.schemas.chunk import DocumentChunk
from app.schemas.retrieval import SearchResult


def make_result(document_id: str, rank: int) -> SearchResult:
    chunk = DocumentChunk(
        id=f"{document_id}:c0",
        document_id=document_id,
        document_version=1,
        chunk_index=0,
        content=f"Content for {document_id}",
        document_title="Test Document",
        document_type="runbook",
        service_id="payment-service",
        source="synthetic",
    )

    return SearchResult(
        chunk=chunk,
        score=1.0 / rank,
        rank=rank,
    )


class FakeRetriever:
    def __init__(self, document_ids: list[str]):
        self.results = [
            make_result(document_id, rank)
            for rank, document_id in enumerate(document_ids, start=1)
        ]

    def search(
        self,
        query: str,
        top_k: int = 5,
        service_id: str | None = None,
    ) -> list[SearchResult]:
        results = self.results

        if service_id is not None:
            results = [
                result for result in results if result.chunk.service_id == service_id
            ]

        return results[:top_k]


def test_rrf_favors_documents_in_both_rankings():
    scores = reciprocal_rank_fusion([["A", "B"], ["B", "C"]])

    assert scores["B"] > scores["A"]
    assert scores["B"] > scores["C"]


def test_rrf_rejects_invalid_k():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["A"]], k=0)


def test_hybrid_returns_unique_documents():
    hybrid = HybridRetriever(
        bm25=FakeRetriever(["A", "B"]),
        semantic=FakeRetriever(["B", "C"]),
    )

    results = hybrid.search("payment timeout")

    ids = [result.chunk.document_id for result in results]

    assert len(ids) == len(set(ids))
    assert set(ids) == {"A", "B", "C"}


def test_hybrid_ranking():
    hybrid = HybridRetriever(
        bm25=FakeRetriever(["A", "B"]),
        semantic=FakeRetriever(["B", "C"]),
    )

    results = hybrid.search("payment timeout")

    assert results[0].chunk.document_id == "B"
    assert results[0].rank == 1


def test_hybrid_top_k():
    hybrid = HybridRetriever(
        bm25=FakeRetriever(["A", "B"]),
        semantic=FakeRetriever(["B", "C"]),
    )

    results = hybrid.search("payment", top_k=2)

    assert len(results) == 2


def test_hybrid_empty_query():
    hybrid = HybridRetriever(
        bm25=FakeRetriever(["A"]),
        semantic=FakeRetriever(["B"]),
    )

    assert hybrid.search("   ") == []


def test_hybrid_invalid_parameters():
    with pytest.raises(ValueError):
        HybridRetriever(
            FakeRetriever([]),
            FakeRetriever([]),
            candidate_k=0,
        )

    with pytest.raises(ValueError):
        HybridRetriever(
            FakeRetriever([]),
            FakeRetriever([]),
            rrf_k=0,
        )


def test_hybrid_invalid_top_k():
    hybrid = HybridRetriever(
        FakeRetriever([]),
        FakeRetriever([]),
    )

    with pytest.raises(ValueError):
        hybrid.search("payment", top_k=0)


def test_hybrid_deterministic():
    hybrid = HybridRetriever(
        bm25=FakeRetriever(["B", "A"]),
        semantic=FakeRetriever(["A", "B"]),
    )

    first = hybrid.search("payment")
    second = hybrid.search("payment")

    assert first == second
