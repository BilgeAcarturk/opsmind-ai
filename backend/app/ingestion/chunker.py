from app.schemas.chunk import DocumentChunk
from app.schemas.knowledge import KnowledgeDocument


def chunk_document(
    document: KnowledgeDocument,
    chunk_size: int = 100,
    overlap: int = 20,
) -> list[DocumentChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    words = document.content.split()

    if not words:
        return []

    step = chunk_size - overlap
    chunks = []

    for start in range(0, len(words), step):
        selected_words = words[start : start + chunk_size]

        if not selected_words:
            break

        chunk_index = len(chunks)

        chunk = DocumentChunk(
            id=(f"{document.id}:v{document.version}:c{chunk_index}"),
            document_id=document.id,
            document_version=document.version,
            chunk_index=chunk_index,
            content=" ".join(selected_words),
            document_title=document.title,
            document_type=document.type,
            service_id=document.service_id,
            source=document.source,
        )

        chunks.append(chunk)

        if start + chunk_size >= len(words):
            break

    return chunks


def chunk_documents(
    documents: list[KnowledgeDocument],
    chunk_size: int = 100,
    overlap: int = 20,
) -> list[DocumentChunk]:
    chunks = []

    for document in documents:
        chunks.extend(
            chunk_document(
                document,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )

    return chunks
