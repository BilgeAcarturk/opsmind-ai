import re

from app.schemas.chunk import DocumentChunk
from app.schemas.retrieval import SearchResult


def tokenize(text: str) -> set[str]:
    """Convert text into a set of lowercase terms."""
    return set(re.findall(r"\b\w+\b", text.lower()))


def keyword_score(query: str, content: str) -> float:
    """Calculate query-term coverage."""
    query_terms = tokenize(query)

    if not query_terms:
        return 0.0

    document_terms = tokenize(content)

    matching_terms = query_terms.intersection(document_terms)

    return len(matching_terms) / len(query_terms)


class KeywordRetriever:
    def __init__(self, chunks: list[DocumentChunk]):
        self._chunks = list(chunks)

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

        scored_chunks = []

        for chunk in self._chunks:
            if service_id is not None and chunk.service_id != service_id:
                continue

            score = keyword_score(query, chunk.content)

            if score > 0:
                scored_chunks.append((chunk, score))

        scored_chunks.sort(key=lambda item: (-item[1], item[0].id))

        results = []

        for rank, (chunk, score) in enumerate(
            scored_chunks[:top_k],
            start=1,
        ):
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    rank=rank,
                )
            )

        return results
