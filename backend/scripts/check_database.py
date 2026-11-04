from app.core.database import get_connection


def main() -> None:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                current_database(),
                current_user,
                (
                    SELECT extversion
                    FROM pg_extension
                    WHERE extname = 'vector'
                )
            """
        ).fetchone()

        print("Database:", row[0])
        print("User:", row[1])
        print("pgvector:", row[2])

        tables = connection.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_type = 'BASE TABLE'
            ORDER BY table_name
            """
        ).fetchall()

        print("\nTables:")

        for table in tables:
            print("-", table[0])


if __name__ == "__main__":
    main()
