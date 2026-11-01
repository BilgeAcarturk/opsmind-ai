
# ADR-0001: Adopt a Modular Monolith Architecture

**Status:** Accepted
**Date:** 2026-10-31

## Context

OpsMind AI requires multiple components:

- HTTP API
- Incident management
- Telemetry retrieval
- Document retrieval
- LLM analysis
- Evaluation
- Observability

The project is initially developed and maintained
by a single developer.

We need clear module boundaries without introducing
unnecessary distributed systems complexity.

## Decision

Use a modular monolith architecture with
explicit boundaries between application components.

The system will initially run as a single
FastAPI backend application.

Components communicate through Python interfaces
and application services.

## Alternatives Considered

### Single-File Application

Advantages:
- Minimal initial setup
- Rapid prototyping

Disadvantages:
- Poor separation of concerns
- Difficult isolated testing
- Increasing maintenance complexity

### Microservices Architecture

Advantages:
- Independent deployment
- Independent scaling
- Strong service isolation

Disadvantages:
- Network communication overhead
- Distributed failure handling
- Increased infrastructure complexity
- More demanding local development

### Modular Monolith

Advantages:
- Clear responsibilities
- Straightforward local development
- Dependency injection
- Isolated testing
- Lower operational complexity

Disadvantages:
- Shared deployment lifecycle
- Requires discipline to maintain boundaries
- Less independent scaling flexibility

## Consequences

Positive:
- Easier testing and debugging
- Faster initial development
- Simpler CI and deployment
- Replaceable infrastructure dependencies

Negative:
- Modules share the same application process
- Future extraction into services may require refactoring

## Reconsideration Criteria

Reevaluate this decision if:

- Components require substantially different scaling.
- Deployment independence becomes necessary.
- Team ownership boundaries change.
- Resource-intensive AI workloads require isolation.

## Related Documents

- docs/PRD.md
- docs/architecture/system-design.md
