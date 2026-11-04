import json
import logging
import sys
from datetime import UTC, datetime

SAFE_FIELDS = (
    "event",
    "request_id",
    "http_method",
    "path",
    "status_code",
    "duration_ms",
    "retrieval_method",
    "top_k",
    "result_count",
    "error_type",
)


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for field in SAFE_FIELDS:
            value = getattr(record, field, None)

            if value is not None:
                payload[field] = value

        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    logger = logging.getLogger("opsmind")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
