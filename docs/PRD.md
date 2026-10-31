
# OpsMind AI — Product Requirements Document

**Version:** 1.0  
**Status:** MVP Scope Defined  
**Project:** AI-Powered Incident Investigation Platform

## 1. Problem Statement

Investigating software incidents requires engineers to
correlate logs, metrics, deployment events, historical
incidents, and operational documentation.

These sources are often distributed across different
systems, making investigation time-consuming.

OpsMind AI aims to assist engineers by collecting
relevant evidence and generating grounded root-cause
hypotheses.

## 2. Target Users

- Site Reliability Engineers (SREs)
- DevOps Engineers
- Backend Engineers

## 3. Product Goals

- Support evidence-based incident investigation.
- Retrieve relevant historical incidents and runbooks.
- Generate root-cause hypotheses linked to evidence.
- Communicate uncertainty when evidence is insufficient.
- Provide measurable AI and retrieval performance.

## 4. MVP Scope

### Included

- Incident listing and detail retrieval
- Incident-specific telemetry inspection
- Retrieval of historical incidents and runbooks
- Evidence-grounded root-cause hypotheses
- Source references and recommended next steps
- Offline evaluation pipeline
- API-level observability

### Excluded

- Autonomous incident remediation
- Automatic production configuration changes
- Predictive incident detection
- Model training from scratch
- Multi-tenant enterprise authentication

## 5. Functional Requirements

| ID | Requirement | Priority |
|---|---|---|
| FR-01 | List available incidents | Must |
| FR-02 | Retrieve incident details | Must |
| FR-03 | Collect relevant telemetry | Must |
| FR-04 | Search historical incidents and runbooks | Must |
| FR-05 | Generate grounded root-cause hypotheses | Must |
| FR-06 | Link claims to supporting evidence | Must |
| FR-07 | Report insufficient evidence | Must |
| FR-08 | Store investigation history | Should |

## 6. Non-Functional Requirements

- Validate API requests and responses.
- Use read-only investigation tools.
- Support automated tests and CI.
- Track retrieval quality and investigation latency.
- Record model and retrieval configuration versions.
- Handle unavailable services and timeouts.
- Avoid exposing credentials or sensitive information.
- Support reproducible local development.

## 7. Success Metrics

### Retrieval

- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)

### Investigation Quality

- Root Cause Top-1 Accuracy
- Root Cause Top-3 Accuracy
- Citation Correctness
- Evidence Faithfulness

### Operational Performance

- p50 and p95 investigation latency
- Cost per investigation
- API error rate

Metric targets will be defined after baseline
experiments and evaluation dataset preparation.

## 8. Data Strategy

Initial development will use synthetic or controlled
incident scenarios with known root causes.

The evaluation ground truth must remain separate
from information accessible to the investigation model.

The project will later integrate real telemetry
from a controlled microservices environment.

## 9. Main User Workflow

1. An engineer selects an incident.
2. The system retrieves incident metadata.
3. Relevant telemetry is collected.
4. Historical documents are retrieved.
5. Evidence is assembled and validated.
6. The LLM generates structured hypotheses.
7. The API returns hypotheses and supporting evidence.
8. The investigation is evaluated and traced.

## 10. Constraints and Risks

- LLMs can generate unsupported conclusions.
- Historical incidents may not match current failures.
- Telemetry may be missing or incomplete.
- Retrieval quality depends on document quality.
- Model latency and cost may limit usability.

The system must not claim a confirmed root cause
without sufficient supporting evidence.

## 11. MVP Acceptance Criteria

- Incident API endpoints are tested.
- Retrieval can be evaluated against labeled examples.
- Investigations return structured responses.
- Evidence references can be validated.
- Missing evidence is handled explicitly.
- CI runs lint, formatting, and automated tests.
- The application can be started reproducibly.
- Evaluation results are documented.

## 12. Future Development

- Live OpenTelemetry integration
- Hybrid retrieval and cross-encoder reranking
- Read-only tool-calling workflows
- Cloud deployment
- Monitoring dashboards
- Advanced incident analysis and reporting
