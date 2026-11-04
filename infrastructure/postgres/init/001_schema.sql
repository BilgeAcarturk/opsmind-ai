
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS knowledge_documents (
    document_id TEXT NOT NULL,
    version INTEGER NOT NULL CHECK (version >= 1),
    document_type TEXT NOT NULL
        CHECK (document_type IN ('runbook', 'postmortem')),
    title TEXT NOT NULL,
    service_id TEXT NOT NULL,
    source TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash CHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (document_id, version)
);

CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id CHAR(64) PRIMARY KEY,
    document_id TEXT NOT NULL,
    document_version INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    config_hash CHAR(64) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    FOREIGN KEY (document_id, document_version)
        REFERENCES knowledge_documents (document_id, version)
        ON DELETE CASCADE,

    UNIQUE (
        document_id,
        document_version,
        config_hash,
        chunk_index
    )
);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id CHAR(64) NOT NULL
        REFERENCES document_chunks (chunk_id)
        ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    embedding vector(384) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (chunk_id, model_id)
);

CREATE INDEX IF NOT EXISTS idx_documents_service
    ON knowledge_documents (service_id);

CREATE INDEX IF NOT EXISTS idx_chunks_document
    ON document_chunks (document_id, document_version);

CREATE INDEX IF NOT EXISTS idx_embeddings_model
    ON chunk_embeddings (model_id);
