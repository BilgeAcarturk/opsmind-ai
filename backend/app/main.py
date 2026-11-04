import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from app.api.health import router as health_router
from app.api.incidents import router as incident_router
from app.api.retrieval import router as retrieval_router
from app.core.database_pool import (
    close_database_pool,
    get_or_create_pool_for_app,
    initialize_pool_state,
)
from app.core.model_provider import get_embedding_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_pool_state(app)

    try:
        if os.getenv("OPSMIND_PRELOAD_ON_STARTUP", "0") == "1":
            await run_in_threadpool(get_embedding_model)

            await run_in_threadpool(
                get_or_create_pool_for_app,
                app,
            )

        yield

    finally:
        await run_in_threadpool(close_database_pool, app)


app = FastAPI(
    title="OpsMind AI",
    description="AI-Powered Incident Investigation Platform",
    version="0.1.0",
    lifespan=lifespan,
)


app.include_router(incident_router)
app.include_router(retrieval_router)
app.include_router(health_router)


@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": "opsmind-api",
        "version": "0.1.0",
    }
