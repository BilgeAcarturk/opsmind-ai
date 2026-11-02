import math
from typing import Protocol

from app.schemas.chunk import DocumentChunk
from app.schemas.retrieval import SearchResult


class Retriever(Protocol):
    def search(
        self,
        query: str,
        top_k: int = 5,
        service_id: str | None = None,
    ) -> list[SearchResult]: ...


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
    weights: list[float] | None = None,
) -> dict[str, float]:
    if k <= 0:
        raise ValueError("RRF k must be positive")

    if weights is None:
        weights = [1.0] * len(rankings)

    if len(weights) != len(rankings):
        raise ValueError("Weights must match rankings")

    if not all(math.isfinite(w) and w >= 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("Weights must be finite and non-negative")

    scores: dict[str, float] = {}

    for ranking, weight in zip(rankings, weights, strict=True):
        seen = set()

        for rank, document_id in enumerate(ranking, start=1):
            if document_id in seen:
                continue

            seen.add(document_id)

            scores[document_id] = scores.get(document_id, 0.0) + weight / (k + rank)

    return scores


class HybridRetriever:
    def __init__(
        self,
        bm25: Retriever,
        semantic: Retriever,
        candidate_k: int = 20,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        semantic_weight: float = 1.0,
    ):
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")

        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")

        weights = [bm25_weight, semantic_weight]

        if not all(math.isfinite(w) and w >= 0 for w in weights) or sum(weights) <= 0:
            raise ValueError("Invalid retrieval weights")

        self._bm25 = bm25
        self._semantic = semantic
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight

    @staticmethod
    def _unique_documents(
        results: list[SearchResult],
    ) -> tuple[list[str], dict[str, DocumentChunk]]:
        document_ids = []
        representatives = {}

        for result in results:
            document_id = result.chunk.document_id

            if document_id not in representatives:
                document_ids.append(document_id)
                representatives[document_id] = result.chunk

        return document_ids, representatives

    def search(
        self,
        query: str,
        top_k: int = 5,
        service_id: str | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if not query.strip():
            return []

        candidate_count = max(top_k, self.candidate_k)

        bm25_results = self._bm25.search(
            query,
            top_k=candidate_count,
            service_id=service_id,
        )

        semantic_results = self._semantic.search(
            query,
            top_k=candidate_count,
            service_id=service_id,
        )

        bm25_ids, bm25_chunks = self._unique_documents(bm25_results)
        semantic_ids, semantic_chunks = self._unique_documents(semantic_results)

        scores = reciprocal_rank_fusion(
            [bm25_ids, semantic_ids],
            k=self.rrf_k,
            weights=[
                self.bm25_weight,
                self.semantic_weight,
            ],
        )

        representatives = {
            **semantic_chunks,
            **bm25_chunks,
        }

        ranked_ids = sorted(
            scores,
            key=lambda document_id: (
                -scores[document_id],
                document_id,
            ),
        )

        return [
            SearchResult(
                chunk=representatives[document_id].model_copy(deep=True),
                score=scores[document_id],
                rank=rank,
            )
            for rank, document_id in enumerate(
                ranked_ids[:top_k],
                start=1,
            )
        ]
