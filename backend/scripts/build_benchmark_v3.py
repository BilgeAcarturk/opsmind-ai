import json
from pathlib import Path

from app.ingestion.document_loader import load_documents
from app.schemas.evaluation import RetrievalDataset
from app.schemas.knowledge import KnowledgeDocument

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "samples"
EVAL_DIR = ROOT / "evaluation" / "datasets"


NEW_DOCUMENTS = [
    {
        "id": "RB-003",
        "type": "runbook",
        "title": "Payment Authentication Failure",
        "service_id": "payment-service",
        "source": "synthetic",
        "content": (
            "Payment requests may fail when API credentials "
            "are invalid, expired, or incorrectly rotated. "
            "Inspect HTTP 401 and 403 responses, authentication "
            "headers, token expiration, and secret versions. "
            "Check whether deployment configuration references "
            "outdated credentials. Credential failures are "
            "different from external provider timeouts."
        ),
        "version": 1,
    },
    {
        "id": "RB-004",
        "type": "runbook",
        "title": "Payment Provider Rate Limiting",
        "service_id": "payment-service",
        "source": "synthetic",
        "content": (
            "Payment gateways may reject requests when API "
            "rate limits are exceeded. Inspect HTTP 429 "
            "responses, request quotas, burst traffic, and "
            "retry-after headers. Aggressive retries may "
            "increase throttling. Apply bounded exponential "
            "backoff and review provider quota policies. "
            "Rate limiting must be distinguished from "
            "authentication failures and network timeouts."
        ),
        "version": 1,
    },
    {
        "id": "RB-005",
        "type": "runbook",
        "title": "Database Connection Leak Investigation",
        "service_id": "order-service",
        "source": "synthetic",
        "content": (
            "Database connections may remain allocated when "
            "application code fails to release resources. "
            "Inspect connection lifetime, active sessions, "
            "pool utilization, and transaction boundaries. "
            "A connection leak can eventually exhaust the "
            "available pool. Identify leaked connections "
            "before increasing the maximum pool size."
        ),
        "version": 1,
    },
    {
        "id": "RB-006",
        "type": "runbook",
        "title": "Network Packet Loss Troubleshooting",
        "service_id": "api-gateway",
        "source": "synthetic",
        "content": (
            "Intermittent packet loss may cause network "
            "retransmissions, connection resets, and elevated "
            "request latency. Inspect network error counters, "
            "TCP retransmissions, dropped packets, and "
            "regional connectivity. Compare affected hosts "
            "and availability zones before changing "
            "application timeout settings."
        ),
        "version": 1,
    },
    {
        "id": "RB-007",
        "type": "runbook",
        "title": "API Gateway Upstream Timeout",
        "service_id": "api-gateway",
        "source": "synthetic",
        "content": (
            "API gateway requests may return HTTP 504 when "
            "upstream services fail to respond before the "
            "gateway deadline. Inspect upstream response "
            "latency, gateway timeout configuration, proxy "
            "logs, and distributed traces. Determine whether "
            "the delay originates in the gateway or a "
            "downstream dependency."
        ),
        "version": 1,
    },
    {
        "id": "RB-008",
        "type": "runbook",
        "title": "Deployment Configuration Regression",
        "service_id": "payment-service",
        "source": "synthetic",
        "content": (
            "An application deployment can introduce "
            "configuration regressions that cause immediate "
            "service errors. Compare environment variables, "
            "feature flags, secrets references, and deployed "
            "image versions. Correlate the error onset with "
            "deployment timestamps. Rollback decisions "
            "require operational approval and supporting "
            "evidence."
        ),
        "version": 1,
    },
]


EVALUATION_CASES = [
    {
        "query_id": "V3-001",
        "query": "payment API keys stopped working after rotation",
        "relevant_document_ids": ["RB-003"],
        "service_id": None,
    },
    {
        "query_id": "V3-002",
        "query": "payment gateway is rejecting too many requests",
        "relevant_document_ids": ["RB-004"],
        "service_id": None,
    },
    {
        "query_id": "V3-003",
        "query": "database sessions are not released by application code",
        "relevant_document_ids": ["RB-005", "RB-002"],
        "service_id": None,
    },
    {
        "query_id": "V3-004",
        "query": "intermittent connectivity and TCP retransmissions",
        "relevant_document_ids": ["RB-006"],
        "service_id": None,
    },
    {
        "query_id": "V3-005",
        "query": "reverse proxy returns 504 waiting for upstream",
        "relevant_document_ids": ["RB-007"],
        "service_id": None,
    },
    {
        "query_id": "V3-006",
        "query": "errors started immediately after a new release",
        "relevant_document_ids": ["RB-008"],
        "service_id": None,
    },
    {
        "query_id": "V3-007",
        "query": "third party payment processor stopped responding",
        "relevant_document_ids": ["RB-001", "PM-001"],
        "service_id": None,
    },
    {
        "query_id": "V3-008",
        "query": "payment requests fail because tokens are expired",
        "relevant_document_ids": ["RB-003"],
        "service_id": "payment-service",
    },
]


def main() -> None:
    original = load_documents(DATA_DIR / "knowledge_documents.json")

    additions = [KnowledgeDocument.model_validate(item) for item in NEW_DOCUMENTS]

    combined = original + additions

    ids = [document.id for document in combined]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate document IDs")

    dataset = RetrievalDataset.model_validate(
        {
            "dataset_version": "0.3.0",
            "label_level": "document",
            "cases": EVALUATION_CASES,
        }
    )

    known_ids = set(ids)
    for case in dataset.cases:
        unknown = set(case.relevant_document_ids) - known_ids
        if unknown:
            raise ValueError(f"{case.query_id}: unknown documents {unknown}")

    document_path = DATA_DIR / "knowledge_documents_v3.json"
    evaluation_path = EVAL_DIR / "retrieval_queries_v3.json"

    document_path.write_text(
        json.dumps(
            [document.model_dump(mode="json") for document in combined],
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    evaluation_path.write_text(
        json.dumps(
            dataset.model_dump(mode="json"),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Documents: {len(combined)}")
    print(f"Queries: {len(dataset.cases)}")
    print(f"Saved: {document_path}")
    print(f"Saved: {evaluation_path}")


if __name__ == "__main__":
    main()
