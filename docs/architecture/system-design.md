
# OpsMind AI — System Design

**Version:** 0.1
**Status:** Proposed Architecture
**Architecture Style:** Modular Monolith

## 1. Overview

OpsMind AI is an AI-powered incident investigation
platform designed to assist engineers in analyzing
software failures using operational evidence and
historical knowledge.

The system combines:

- Structured telemetry queries
- Historical incident retrieval
- Semantic and keyword search
- Evidence-grounded LLM analysis
- Automated evaluation and observability

## 2. Architecture Principles

1. Separate HTTP handling from business logic.
2. Keep data access behind repository interfaces.
3. Treat LLMs as replaceable dependencies.
4. Require traceable evidence for generated claims.
5. Separate evaluation ground truth from model inputs.
6. Prefer deterministic workflows before autonomous agents.
7. Measure quality, latency, and cost independently.
8. Restrict investigation tools to read-only operations.

## 3. Main Components

### 3.1 API Layer

Responsibilities:
- HTTP request handling
- Input and output validation
- Error responses
- Dependency injection

Technology: FastAPI and Pydantic.

### 3.2 Incident Management

Responsibilities:
- Incident listing and lookup
- Incident metadata management
- Service and incident relationships

Current storage: In-memory repository.

Planned storage: PostgreSQL.

### 3.3 Investigation Orchestrator

Responsibilities:
- Coordinate investigation steps
- Query telemetry sources
- Trigger historical knowledge retrieval
- Assemble analysis context
- Invoke the LLM analysis service
- Validate and store investigation results

The initial workflow will be deterministic.

### 3.4 Telemetry Service

Responsibilities:
- Query incident-related logs
- Retrieve metrics and trace records
- Filter data by service and time range
- Return structured operational evidence

The initial implementation will use sample data.

### 3.5 Retrieval Service

Responsibilities:
- Index historical incident documents
- Index runbooks and postmortems
- Perform keyword and semantic retrieval
- Filter results using metadata
- Rerank retrieved candidates

Planned storage: PostgreSQL with pgvector.

### 3.6 Evidence Builder

Responsibilities:
- Normalize retrieved evidence
- Attach stable evidence identifiers
- Preserve source references
- Remove duplicates
- Enforce context size constraints

### 3.7 LLM Analysis Service

Responsibilities:
- Generate root-cause hypotheses
- Produce structured investigation summaries
- Reference supporting evidence
- Recommend further investigation steps
- Express uncertainty

The system must not treat LLM output as verified fact.

### 3.8 Response Validator

Responsibilities:
- Validate response schemas
- Verify evidence identifiers exist
- Detect missing required fields
- Reject structurally invalid responses

Semantic correctness requires separate evaluation.

## 4. Investigation Workflow

1. Client submits an incident investigation request.
2. API validates the request.
3. Incident Service retrieves incident metadata.
4. Orchestrator requests telemetry evidence.
5. Retrieval Service searches historical knowledge.
6. Evidence Builder constructs a bounded context.
7. LLM Analysis Service generates hypotheses.
8. Response Validator checks structural correctness.
9. Investigation results are returned.
10. Evaluation and observability record execution data.

## 5. Data Storage

### Operational Database

Stores:
- Services
- Incidents
- Investigation metadata
- Evidence references

### Knowledge Store

Stores:
- Runbooks
- Historical incidents
- Postmortems
- Document chunks and embeddings

### Telemetry Store

Stores or provides access to:
- Logs
- Metrics
- Traces
- Deployment events

These are separate logical responsibilities.
Physical storage may be shared in the MVP.

## 6. Reliability and Safety

- External requests must have timeouts.
- Unavailable dependencies must produce controlled errors.
- Investigation tools must be read-only.
- Secrets must not be committed to the repository.
- Sensitive telemetry fields must be sanitized.
- Missing evidence must be reported explicitly.
- LLM-generated instructions must not execute automatically.

## 7. Evaluation

Offline evaluation will use versioned datasets
containing known incident root causes and
relevant evidence labels.

Evaluation metrics include:

- Precision@K
- Recall@K
- MRR
- Root Cause Top-1 / Top-3 Accuracy
- Citation Correctness
- Faithfulness
- Latency
- Cost per investigation

Ground truth must not enter the LLM context.

## 8. Current Implementation

Implemented:
- FastAPI application
- Pydantic domain schemas
- Incident Repository and Service layers
- Incident API endpoints
- Automated tests
- GitHub Actions CI

Planned:
- Telemetry ingestion
- Knowledge indexing
- Hybrid retrieval
- LLM investigation pipeline
- Evaluation framework
- Monitoring and deployment

## 9. Evolution Strategy

Phase 1: In-memory incident management.
Phase 2: Sample telemetry and document ingestion.
Phase 3: Retrieval engineering and evaluation.
Phase 4: LLM investigation pipeline.
Phase 5: Observability and containerized deployment.
Phase 6: Live telemetry integration.
