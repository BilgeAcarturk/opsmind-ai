import logging
import time
from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.retrieval_dependencies import get_postgres_retriever
from app.retrieval.postgres_semantic import (
    PostgresSemanticRetriever,
)
from app.schemas.retrieval_api import (
    RetrievalSearchRequest,
    RetrievalSearchResponse,
)

logger = logging.getLogger("opsmind.retrieval")

router = APIRouter(
    prefix="/api/v1/retrieval",
    tags=["Retrieval"],
)


@router.post(
    "/search",
    response_model=RetrievalSearchResponse,
)
def search_documents(
    request: RetrievalSearchRequest,
    http_request: Request,
    retriever: Annotated[
        PostgresSemanticRetriever,
        Depends(get_postgres_retriever),
    ],
) -> RetrievalSearchResponse:
    request_id = getattr(
        http_request.state,
        "request_id",
        None,
    )

    started = time.perf_counter()

    try:
        results = retriever.search(
            query=request.query,
            top_k=request.top_k,
            service_id=request.service_id,
        )

    except psycopg.Error as exc:
        logger.error(
            "Retrieval failed",
            extra={
                "event": "retrieval_failed",
                "request_id": request_id,
                "retrieval_method": "postgres_semantic",
                "error_type": type(exc).__name__,
            },
        )

        raise HTTPException(
            status_code=503,
            detail="Retrieval service temporarily unavailable",
        ) from None

    duration_ms = (time.perf_counter() - started) * 1000

    logger.info(
        "Retrieval completed",
        extra={
            "event": "retrieval_completed",
            "request_id": request_id,
            "retrieval_method": "postgres_semantic",
            "top_k": request.top_k,
            "result_count": len(results),
            "duration_ms": round(duration_ms, 3),
        },
    )

    return RetrievalSearchResponse(
        query=request.query,
        method="postgres_semantic",
        total_results=len(results),
        results=results,
    )
