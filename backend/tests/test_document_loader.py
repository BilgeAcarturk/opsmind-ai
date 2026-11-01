import json

import pytest
from pydantic import ValidationError

from app.ingestion.document_loader import load_documents


def test_load_valid_documents(tmp_path):
    documents = [
        {
            "id": "RB-TEST",
            "type": "runbook",
            "title": "Payment Timeout Guide",
            "service_id": "payment-service",
            "source": "synthetic",
            "content": "Inspect payment provider latency and errors.",
            "version": 1,
        }
    ]

    file_path = tmp_path / "documents.json"
    file_path.write_text(json.dumps(documents), encoding="utf-8")

    result = load_documents(file_path)

    assert len(result) == 1
    assert result[0].id == "RB-TEST"


def test_duplicate_document_ids(tmp_path):
    document = {
        "id": "RB-001",
        "type": "runbook",
        "title": "Payment Timeout Guide",
        "service_id": "payment-service",
        "source": "synthetic",
        "content": "Inspect payment provider latency and errors.",
        "version": 1,
    }

    file_path = tmp_path / "duplicates.json"
    file_path.write_text(
        json.dumps([document, document]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate"):
        load_documents(file_path)


def test_invalid_document_type(tmp_path):
    documents = [
        {
            "id": "DOC-001",
            "type": "unknown",
            "title": "Invalid Document",
            "service_id": "payment-service",
            "source": "synthetic",
            "content": "This is an invalid document type.",
            "version": 1,
        }
    ]

    file_path = tmp_path / "invalid.json"
    file_path.write_text(json.dumps(documents), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_documents(file_path)
