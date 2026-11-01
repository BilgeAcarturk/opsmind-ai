import json
from pathlib import Path

from pydantic import TypeAdapter

from app.schemas.knowledge import KnowledgeDocument

_document_list_adapter = TypeAdapter(list[KnowledgeDocument])


def load_documents(
    file_path: str | Path,
) -> list[KnowledgeDocument]:
    path = Path(file_path)

    with path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    documents = _document_list_adapter.validate_python(raw_data)

    ids = [document.id for document in documents]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate document IDs detected")

    return documents
