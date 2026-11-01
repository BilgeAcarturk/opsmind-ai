import numpy as np
import pytest

from app.retrieval.semantic import SemanticRetriever
from app.schemas.chunk import DocumentChunk


class FakeEmbeddingModel:
    def encode(
        self,
        sentences,
        *,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ):
        vectors = {
            "payment failure": [1.0, 0.0],
            "database failure": [0.0, 1.0],
            "payment issue": [1.0, 0.0],
            "database issue": [0.0, 1.0],
        }
        return np.asarray(
            [vectors[text] for text in sentences],
            dtype=np.float32,
        )


def make_chunk(chunk_id, content, service_id):
    return DocumentChunk(
        id=chunk_id,
        document_id=chunk_id,
        document_version=1,
        chunk_index=0,
        content=content,
        document_title="Test Document",
        document_type="runbook",
        service_id=service_id,
        source="synthetic",
    )


@pytest.fixture
def retriever():
    chunks = [
        make_chunk("A", "payment failure", "payment-service"),
        make_chunk("B", "database failure", "database-service"),
    ]
    return SemanticRetriever(chunks, FakeEmbeddingModel())


def test_semantic_ranking(retriever):
    results = retriever.search("payment issue")

    assert results[0].chunk.id == "A"
    assert results[0].score == pytest.approx(1.0)


def test_semantic_service_filter(retriever):
    results = retriever.search(
        "payment issue",
        service_id="database-service",
    )

    assert len(results) == 1
    assert results[0].chunk.id == "B"


def test_semantic_top_k(retriever):
    results = retriever.search("database issue", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.id == "B"


def test_empty_semantic_query(retriever):
    assert retriever.search("   ") == []


def test_invalid_semantic_top_k(retriever):
    with pytest.raises(ValueError):
        retriever.search("payment issue", top_k=0)


def test_empty_semantic_index():
    retriever = SemanticRetriever([], FakeEmbeddingModel())

    assert retriever.search("payment issue") == []
