from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from app.schemas.chunk import DocumentChunk
from app.schemas.retrieval import SearchResult


class EmbeddingModel(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> NDArray[np.float32]: ...


class SemanticRetriever:
    def __init__(
        self,
        chunks: list[DocumentChunk],
        model: EmbeddingModel,
    ):
        self._chunks = [chunk.model_copy(deep=True) for chunk in chunks]
        self._model = model

        if not self._chunks:
            self._embeddings = np.empty((0, 0))
            return

        self._embeddings = np.asarray(
            model.encode(
                [chunk.content for chunk in self._chunks],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        )

        if self._embeddings.ndim != 2 or self._embeddings.shape[0] != len(self._chunks):
            raise ValueError("Invalid document embedding shape")

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

        query_embedding = np.asarray(
            self._model.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        )

        if query_embedding.shape != (
            1,
            self._embeddings.shape[1],
        ):
            raise ValueError("Invalid query embedding shape")

        scores = self._embeddings @ query_embedding[0]

        candidates = []

        for chunk, score in zip(self._chunks, scores, strict=True):
            if service_id is not None and chunk.service_id != service_id:
                continue

            candidates.append((chunk, float(score)))

        candidates.sort(key=lambda item: (-item[1], item[0].id))

        return [
            SearchResult(
                chunk=chunk,
                score=max(-1.0, min(1.0, score)),
                rank=rank,
            )
            for rank, (chunk, score) in enumerate(
                candidates[:top_k],
                start=1,
            )
        ]
