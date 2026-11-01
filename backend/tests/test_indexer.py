import pytest

from app.ingestion.indexer import build_index
from app.schemas.indexing import IndexingConfig
from app.schemas.knowledge import KnowledgeDocument


def make_document(document_id="RB-001"):
    return KnowledgeDocument(
        id=document_id,
        type="runbook",
        title="Payment Timeout Guide",
        service_id="payment-service",
        source="synthetic",
        content=" ".join(["payment timeout latency"] * 30),
        version=1,
    )


def test_index_is_deterministic():
    documents = [make_document()]
    config = IndexingConfig(chunk_size=20, overlap=5)

    chunks_a, manifest_a = build_index(documents, config)
    chunks_b, manifest_b = build_index(documents, config)

    assert [c.id for c in chunks_a] == [c.id for c in chunks_b]
    assert manifest_a.dataset_hash == manifest_b.dataset_hash
    assert manifest_a.config_hash == manifest_b.config_hash
    assert manifest_a.run_id != manifest_b.run_id


def test_config_changes_chunk_identity():
    documents = [make_document()]

    chunks_a, manifest_a = build_index(
        documents, IndexingConfig(chunk_size=20, overlap=5)
    )
    chunks_b, manifest_b = build_index(
        documents, IndexingConfig(chunk_size=30, overlap=5)
    )

    assert manifest_a.config_hash != manifest_b.config_hash
    assert chunks_a[0].id != chunks_b[0].id


def test_document_change_updates_identity():
    first = make_document()
    updated = first.model_copy(update={"content": first.content + " provider error"})
    config = IndexingConfig(chunk_size=20, overlap=5)

    chunks_a, manifest_a = build_index([first], config)
    chunks_b, manifest_b = build_index([updated], config)

    assert manifest_a.dataset_hash != manifest_b.dataset_hash
    assert chunks_a[0].id != chunks_b[0].id


def test_duplicate_documents_are_rejected():
    document = make_document()

    with pytest.raises(ValueError, match="Duplicate"):
        build_index([document, document], IndexingConfig())


def test_invalid_indexing_config():
    with pytest.raises(ValueError):
        IndexingConfig(chunk_size=10, overlap=10)


def test_document_order_does_not_change_index():
    first = make_document("RB-001")
    second = make_document("RB-002")
    config = IndexingConfig()

    chunks_a, manifest_a = build_index([first, second], config)
    chunks_b, manifest_b = build_index([second, first], config)

    assert [c.id for c in chunks_a] == [c.id for c in chunks_b]
    assert manifest_a.dataset_hash == manifest_b.dataset_hash
