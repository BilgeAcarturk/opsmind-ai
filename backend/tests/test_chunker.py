import pytest

from app.ingestion.chunker import chunk_document, chunk_documents
from app.schemas.knowledge import KnowledgeDocument


def make_document(content: str) -> KnowledgeDocument:
    return KnowledgeDocument(
        id="RB-TEST",
        type="runbook",
        title="Test Runbook",
        service_id="payment-service",
        source="synthetic",
        content=content,
        version=1,
    )


def test_short_document_creates_one_chunk():
    document = make_document(" ".join(["payment"] * 30))

    chunks = chunk_document(
        document,
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) == 1
    assert len(chunks[0].content.split()) == 30
    assert chunks[0].chunk_index == 0


def test_chunk_overlap():
    words = [f"word{i}" for i in range(20)]
    document = make_document(" ".join(words))

    chunks = chunk_document(
        document,
        chunk_size=10,
        overlap=3,
    )

    assert len(chunks) == 3

    first = chunks[0].content.split()
    second = chunks[1].content.split()

    assert first[-3:] == second[:3]


def test_chunk_metadata_is_preserved():
    document = make_document(" ".join(["timeout"] * 30))

    chunk = chunk_document(
        document,
        chunk_size=20,
        overlap=5,
    )[0]

    assert chunk.document_id == "RB-TEST"
    assert chunk.document_version == 1
    assert chunk.service_id == "payment-service"
    assert chunk.document_type.value == "runbook"
    assert chunk.source == "synthetic"


def test_invalid_chunk_parameters():
    document = make_document(" ".join(["timeout"] * 30))

    with pytest.raises(ValueError):
        chunk_document(document, chunk_size=0)

    with pytest.raises(ValueError):
        chunk_document(document, chunk_size=10, overlap=10)

    with pytest.raises(ValueError):
        chunk_document(document, chunk_size=10, overlap=-1)


def test_chunking_is_deterministic():
    document = make_document(" ".join(["payment"] * 150))

    first = chunk_document(document)
    second = chunk_document(document)

    assert first == second


def test_chunk_multiple_documents():
    first = make_document(" ".join(["payment"] * 30))

    second = first.model_copy(update={"id": "RB-SECOND"})

    chunks = chunk_documents(
        [first, second],
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) == 2
    assert chunks[0].document_id == "RB-TEST"
    assert chunks[1].document_id == "RB-SECOND"
