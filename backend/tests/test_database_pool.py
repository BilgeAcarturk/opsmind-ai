from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.core.database_pool import (
    close_database_pool,
    get_or_create_pool,
    initialize_pool_state,
)


def make_request():
    app = SimpleNamespace(state=SimpleNamespace())
    initialize_pool_state(app)

    request = SimpleNamespace(app=app)
    return request


def test_pool_created_once(monkeypatch):
    request = make_request()
    fake_pool = MagicMock()

    factory = MagicMock(return_value=fake_pool)

    monkeypatch.setattr(
        "app.core.database_pool.create_database_pool",
        factory,
    )

    first = get_or_create_pool(request)
    second = get_or_create_pool(request)

    assert first is second
    factory.assert_called_once()
    fake_pool.open.assert_called_once_with(
        wait=True,
        timeout=5.0,
    )

    close_database_pool(request.app)


def test_pool_closed_on_shutdown(monkeypatch):
    request = make_request()
    fake_pool = MagicMock()

    monkeypatch.setattr(
        "app.core.database_pool.create_database_pool",
        lambda: fake_pool,
    )

    get_or_create_pool(request)
    close_database_pool(request.app)

    fake_pool.close.assert_called_once()
    assert request.app.state.db_pool is None


def test_pool_not_created_until_needed():
    request = make_request()

    assert request.app.state.db_pool is None

    close_database_pool(request.app)


def test_failed_pool_open_is_cleaned_up(monkeypatch):
    request = make_request()
    fake_pool = MagicMock()
    fake_pool.open.side_effect = RuntimeError("Database unavailable")

    monkeypatch.setattr(
        "app.core.database_pool.create_database_pool",
        lambda: fake_pool,
    )

    with pytest.raises(RuntimeError):
        get_or_create_pool(request)

    fake_pool.close.assert_called_once()
    assert request.app.state.db_pool is None
