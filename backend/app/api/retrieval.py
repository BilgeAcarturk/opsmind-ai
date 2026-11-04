import logging
from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends, HTTPException

from app.api.retrieval_dependencies import get_postgres_retriever
from app.retrieval.postgres_semantic import (
    PostgresSemanticRetriever,
)
from app.schemas.retrieval_api import (
    RetrievalSearchRequest,
    RetrievalSearchResponse,
)

logger = logging.getLogger(__name__)

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
    retriever: Annotated[
        PostgresSemanticRetriever,
        Depends(get_postgres_retriever),
    ],
) -> RetrievalSearchResponse:
    try:
        results = retriever.search(
            query=request.query,
            top_k=request.top_k,
            service_id=request.service_id,
        )

    except psycopg.Error:
        logger.exception("PostgreSQL retrieval failed")

        raise HTTPException(
            status_code=503,
            detail="Retrieval service temporarily unavailable",
        ) from None

    return RetrievalSearchResponse(
        query=request.query,
        method="postgres_semantic",
        total_results=len(results),
        results=results,
    )
