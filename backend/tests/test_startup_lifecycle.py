from importlib import import_module
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

main_module = import_module("app.main")


def test_default_startup_is_lazy(monkeypatch):
    monkeypatch.delenv(
        "OPSMIND_PRELOAD_ON_STARTUP",
        raising=False,
    )

    model_loader = MagicMock()
    pool_factory = MagicMock()

    monkeypatch.setattr(
        main_module,
        "get_embedding_model",
        model_loader,
    )
    monkeypatch.setattr(
        main_module,
        "get_or_create_pool_for_app",
        pool_factory,
    )

    with TestClient(main_module.app) as client:
        response = client.get("/health/live")
        assert response.status_code == 200

    model_loader.assert_not_called()
    pool_factory.assert_not_called()


def test_warm_startup_prepares_resources(monkeypatch):
    monkeypatch.setenv(
        "OPSMIND_PRELOAD_ON_STARTUP",
        "1",
    )

    model_loader = MagicMock()
    pool_factory = MagicMock()

    monkeypatch.setattr(
        main_module,
        "get_embedding_model",
        model_loader,
    )
    monkeypatch.setattr(
        main_module,
        "get_or_create_pool_for_app",
        pool_factory,
    )

    with TestClient(main_module.app) as client:
        assert client.get("/health/live").status_code == 200

    model_loader.assert_called_once()
    pool_factory.assert_called_once_with(main_module.app)


def test_startup_failure_prevents_serving(monkeypatch):
    monkeypatch.setenv(
        "OPSMIND_PRELOAD_ON_STARTUP",
        "1",
    )

    def failing_model_loader():
        raise RuntimeError("Model unavailable")

    pool_factory = MagicMock()

    monkeypatch.setattr(
        main_module,
        "get_embedding_model",
        failing_model_loader,
    )
    monkeypatch.setattr(
        main_module,
        "get_or_create_pool_for_app",
        pool_factory,
    )

    with (
        pytest.raises(
            RuntimeError,
            match="Model unavailable",
        ),
        TestClient(main_module.app),
    ):
        pass

    pool_factory.assert_not_called()
