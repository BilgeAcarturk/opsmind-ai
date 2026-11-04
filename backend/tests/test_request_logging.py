import json
import logging
from io import StringIO
from unittest.mock import patch
from uuid import UUID

from fastapi.testclient import TestClient

from app.core.logging_config import JSONFormatter
from app.main import app


def test_json_formatter_produces_valid_json():
    formatter = JSONFormatter()

    record = logging.LogRecord(
        name="opsmind.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Test completed",
        args=(),
        exc_info=None,
    )
    record.event = "test_completed"
    record.request_id = "test-123"

    parsed = json.loads(formatter.format(record))

    assert parsed["level"] == "INFO"
    assert parsed["event"] == "test_completed"
    assert parsed["request_id"] == "test-123"
    assert "timestamp" in parsed


def test_request_id_header_is_valid_uuid():
    with TestClient(app) as client:
        response = client.get("/health/live")

    assert response.status_code == 200
    UUID(response.headers["X-Request-ID"])


def test_each_request_gets_unique_id():
    with TestClient(app) as client:
        first = client.get("/health/live")
        second = client.get("/health/live")

    assert first.headers["X-Request-ID"] != second.headers["X-Request-ID"]


def test_http_log_contains_safe_metadata():
    captured = StringIO()
    handler = logging.StreamHandler(captured)
    handler.setFormatter(JSONFormatter())

    logger = logging.getLogger("opsmind.http")
    logger.addHandler(handler)

    try:
        with TestClient(app) as client:
            response = client.get("/health/live")
    finally:
        logger.removeHandler(handler)
        handler.close()

    lines = captured.getvalue().splitlines()
    assert lines

    event = json.loads(lines[-1])

    assert event["event"] == "http_request_completed"
    assert event["status_code"] == 200
    assert event["path"] == "/health/live"
    assert event["duration_ms"] >= 0
    assert event["request_id"] == response.headers["X-Request-ID"]


def test_request_log_does_not_contain_query_text():

    with (
        patch("app.core.request_logging.logger.info") as log,
        TestClient(app) as client,
    ):
        client.get("/health/live?secret=do-not-log-this")

    assert log.called

    for call in log.call_args_list:
        assert "do-not-log-this" not in str(call)
