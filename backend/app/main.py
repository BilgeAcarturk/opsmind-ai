from fastapi import FastAPI

from app.api.incidents import router as incident_router

app = FastAPI(
    title="OpsMind AI",
    description="AI-Powered Incident Investigation Platform",
    version="0.1.0",
)

app.include_router(incident_router)


@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": "opsmind-api",
        "version": "0.1.0",
    }
