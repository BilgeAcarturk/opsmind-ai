import math
from collections import Counter

from app.retrieval.keyword import tokenize
from app.schemas.chunk import DocumentChunk
from app.schemas.retrieval import SearchResult


class BM25Retriever:
    def __init__(
        self,
        chunks: list[DocumentChunk],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if k1 <= 0:
            raise ValueError("k1 must be positive")

        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self.k1 = k1
        self.b = b
        self._chunks = [chunk.model_copy(deep=True) for chunk in chunks]

        self._term_frequencies = []
        self._document_frequencies = Counter()
        self._document_lengths = []

        for chunk in self._chunks:
            tokens = self._tokenize_with_counts(chunk.content)
            frequencies = Counter(tokens)

            self._term_frequencies.append(frequencies)
            self._document_frequencies.update(frequencies.keys())
            self._document_lengths.append(len(tokens))

        self._document_count = len(self._chunks)

        self._average_length = (
            sum(self._document_lengths) / self._document_count
            if self._document_count
            else 0.0
        )

    @staticmethod
    def _tokenize_with_counts(text: str) -> list[str]:
        import re

        return re.findall(r"\b\w+\b", text.lower())

    def _idf(self, term: str) -> float:
        document_frequency = self._document_frequencies.get(term, 0)

        numerator = self._document_count - document_frequency + 0.5
        denominator = document_frequency + 0.5

        return math.log1p(numerator / denominator)

    def _score(
        self,
        query_terms: set[str],
        chunk_index: int,
    ) -> float:
        frequencies = self._term_frequencies[chunk_index]
        document_length = self._document_lengths[chunk_index]

        if self._average_length == 0:
            return 0.0

        score = 0.0

        for term in query_terms:
            frequency = frequencies.get(term, 0)

            if frequency == 0:
                continue

            length_factor = 1 - self.b + self.b * document_length / self._average_length

            numerator = frequency * (self.k1 + 1)
            denominator = frequency + self.k1 * length_factor

            score += self._idf(term) * numerator / denominator

        return score

    def search(
        self,
        query: str,
        top_k: int = 5,
        service_id: str | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if not query.strip() or not self._chunks:
            return []

        query_terms = tokenize(query)
        candidates = []

        for index, chunk in enumerate(self._chunks):
            if service_id is not None and chunk.service_id != service_id:
                continue

            score = self._score(query_terms, index)

            if score > 0:
                candidates.append((chunk, score))

        candidates.sort(key=lambda item: (-item[1], item[0].id))

        return [
            SearchResult(
                chunk=chunk,
                score=score,
                rank=rank,
            )
            for rank, (chunk, score) in enumerate(
                candidates[:top_k],
                start=1,
            )
        ]
