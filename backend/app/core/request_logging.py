import logging
import time
from uuid import uuid4

from fastapi import Request

logger = logging.getLogger("opsmind.http")


async def request_logging_middleware(
    request: Request,
    call_next,
):
    request_id = str(uuid4())
    request.state.request_id = request_id

    started = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response

    finally:
        duration_ms = (time.perf_counter() - started) * 1000

        logger.info(
            "HTTP request finished",
            extra={
                "event": "http_request_completed",
                "request_id": request_id,
                "http_method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "duration_ms": round(duration_ms, 3),
            },
        )
