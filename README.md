
# OpsMind AI

AI-Powered Incident Investigation & Cloud Operations
Intelligence Platform.

## Overview

OpsMind AI is an AI engineering project designed to
support incident investigation using operational
telemetry, historical incidents, and runbooks.

## Planned Capabilities

- Incident management API
- Telemetry evidence retrieval
- Hybrid semantic and keyword search
- Evidence-grounded root cause hypotheses
- Retrieval and LLM evaluation
- Observability and deployment

## Tech Stack

- Python 3.12
- FastAPI
- Pydantic
- pytest
- Ruff
- uv

## Getting Started

```bash
cd backend
uv sync --dev
uv run uvicorn app.main:app --reload
```

API documentation:

http://127.0.0.1:8000/docs

## Running Tests

```bash
cd backend
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
```

## Project Status

Milestone 1 — Backend foundation in progress.

## License

To be determined.
