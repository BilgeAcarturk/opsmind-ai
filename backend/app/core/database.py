import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

PROJECT_ROOT = Path(__file__).resolve().parents[3]

load_dotenv(PROJECT_ROOT / ".env")


def get_connection() -> psycopg.Connection:
    password = os.getenv("POSTGRES_PASSWORD")

    if not password:
        raise RuntimeError("POSTGRES_PASSWORD is missing")

    connection = psycopg.connect(
        host="127.0.0.1",
        port=5433,
        dbname="opsmind",
        user="opsmind",
        password=password,
        connect_timeout=5,
    )

    try:
        register_vector(connection)
    except Exception:
        connection.close()
        raise

    return connection
