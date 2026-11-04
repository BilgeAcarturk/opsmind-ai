import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

PROJECT_ROOT = Path(__file__).resolve().parents[3]

load_dotenv(PROJECT_ROOT / ".env")


def get_database_kwargs() -> dict:
    password = os.getenv("POSTGRES_PASSWORD")

    if not password:
        raise RuntimeError("POSTGRES_PASSWORD is missing")

    return {
        "host": os.getenv("POSTGRES_HOST", "127.0.0.1"),
        "port": int(os.getenv("POSTGRES_PORT", "5433")),
        "dbname": os.getenv("POSTGRES_DB", "opsmind"),
        "user": os.getenv("POSTGRES_USER", "opsmind"),
        "password": password,
        "connect_timeout": 5,
    }


def configure_vector_connection(
    connection: psycopg.Connection,
) -> None:
    register_vector(connection)

    # pgvector registration may execute database queries.
    # Reset the transaction before the pool uses this connection.
    connection.rollback()


def get_connection() -> psycopg.Connection:
    connection = psycopg.connect(**get_database_kwargs())

    try:
        configure_vector_connection(connection)
    except Exception:
        connection.close()
        raise

    return connection
