from threading import Lock

from fastapi import Request
from psycopg_pool import ConnectionPool

from app.core.database import (
    configure_vector_connection,
    get_database_kwargs,
)


def create_database_pool() -> ConnectionPool:
    return ConnectionPool(
        kwargs=get_database_kwargs(),
        min_size=1,
        max_size=5,
        timeout=5.0,
        configure=configure_vector_connection,
        open=False,
    )


def get_or_create_pool(request: Request) -> ConnectionPool:
    app = request.app

    with app.state.db_pool_lock:
        pool = app.state.db_pool

        if pool is None:
            pool = create_database_pool()

            try:
                pool.open(wait=True, timeout=5.0)
            except Exception:
                pool.close()
                raise

            app.state.db_pool = pool

        return pool


def initialize_pool_state(app) -> None:
    app.state.db_pool = None
    app.state.db_pool_lock = Lock()


def close_database_pool(app) -> None:
    with app.state.db_pool_lock:
        pool = app.state.db_pool
        app.state.db_pool = None

        if pool is not None:
            pool.close()
