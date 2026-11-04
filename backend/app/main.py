from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.incidents import router as incident_router
from app.api.retrieval import router as retrieval_router
from app.core.database_pool import (
    close_database_pool,
    initialize_pool_state,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_pool_state(app)

    try:
        yield
    finally:
        close_database_pool(app)


app = FastAPI(
    title="OpsMind AI",
    description="AI-Powered Incident Investigation Platform",
    version="0.1.0",
    lifespan=lifespan,
)


app.include_router(incident_router)
app.include_router(retrieval_router)


@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": "opsmind-api",
        "version": "0.1.0",
    }
