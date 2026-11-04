import logging

import psycopg
from fastapi import APIRouter, HTTPException, Request
from psycopg_pool import PoolClosed, PoolTimeout

from app.core.model_provider import get_embedding_model

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


@router.get("/live")
def liveness():
    return {"status": "alive"}


@router.get("/ready")
def readiness(request: Request):
    model_loaded = get_embedding_model.cache_info().currsize > 0

    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Embedding model is not loaded",
        )

    pool = getattr(
        request.app.state,
        "db_pool",
        None,
    )

    if pool is None or pool.closed:
        raise HTTPException(
            status_code=503,
            detail="Database pool is not ready",
        )

    try:
        with pool.connection(timeout=1.0) as connection:
            row = connection.execute("SELECT 1").fetchone()

            if row is None or row[0] != 1:
                raise HTTPException(
                    status_code=503,
                    detail="Database readiness check failed",
                )

    except (psycopg.Error, PoolTimeout, PoolClosed):
        logger.exception("Database readiness check failed")

        raise HTTPException(
            status_code=503,
            detail="Database unavailable",
        ) from None

    return {
        "status": "ready",
        "model": "loaded",
        "database": "available",
    }
