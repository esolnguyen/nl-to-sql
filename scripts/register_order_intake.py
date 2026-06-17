"""Register the Quilbyte order-intake Postgres DB as an nl-to-sql connection.

This mirrors what ``DatabaseConnectionService.create`` does (encrypt the URI,
detect the dialect, persist a ``DatabaseConnection`` plus one ``NOT_SCANNED``
``TableDescription`` per table) but without importing the heavy scanner chain
(snowflake/clickhouse/etc.), so it runs with a minimal dependency set.

It does NOT run the full schema scan (cardinality, sample values) — that step
needs an OpenAI key and is triggered later via POST
``/api/v1/table-descriptions/sync-schemas``.

Usage:
    PYTHONPATH=src python scripts/register_order_intake.py
"""
import os
import sys

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

load_dotenv()

from config import Settings, System  # noqa: E402
from databases.mongodb.mongo import MongoDB  # noqa: E402
from models.db_connection import DatabaseConnection  # noqa: E402
from models.db_description import TableDescription, TableDescriptionStatus  # noqa: E402
from repositories.db_connections import DatabaseConnectionRepository  # noqa: E402
from repositories.table_descriptions import TableDescriptionRepository  # noqa: E402

ALIAS = "quilbyte-order-intake"
SCHEMA = "public"
# No credential is hard-coded here: the target URI must come from the
# (git-ignored) .env via ORDER_INTAKE_DATABASE_URL.


def list_tables(uri: str) -> list[str]:
    engine = create_engine(uri)
    try:
        with engine.connect():
            pass
        inspector = inspect(engine)
        tables = inspector.get_table_names(schema=SCHEMA) + inspector.get_view_names(
            schema=SCHEMA
        )
    finally:
        engine.dispose()
    if not tables:
        raise SystemExit("No tables found — check the connection URI / permissions.")
    return sorted({t.lower() for t in tables})


def main() -> None:
    uri = os.getenv("ORDER_INTAKE_DATABASE_URL")
    if not uri:
        raise SystemExit(
            "ORDER_INTAKE_DATABASE_URL is not set. Add it to .env, e.g.\n"
            "  ORDER_INTAKE_DATABASE_URL=postgresql://<user>:<pass>@<host>:<port>/<db>"
        )
    print("→ Connecting to target DB and listing tables...")
    tables = list_tables(uri)
    print(f"  found {len(tables)} tables/views in schema '{SCHEMA}'.")

    system = System(Settings())
    storage = MongoDB(system)
    conn_repo = DatabaseConnectionRepository(storage)
    table_repo = TableDescriptionRepository(storage)

    existing = conn_repo.find_one({"alias": ALIAS})
    if existing:
        db_connection = existing
        print(f"→ Connection '{ALIAS}' already exists: {db_connection.id} (reusing).")
    else:
        # The connection_uri validator encrypts the URI and sets the dialect.
        db_connection = DatabaseConnection(
            alias=ALIAS,
            connection_uri=uri,
            schemas=[SCHEMA],
            metadata={"source": "register_order_intake.py"},
        )
        db_connection = conn_repo.insert(db_connection)
        print(
            f"→ Registered connection '{ALIAS}': {db_connection.id} "
            f"(dialect={db_connection.dialect})."
        )

    created = 0
    for table in tables:
        if table_repo.get_table_info(db_connection.id, table):
            continue
        table_repo.save_table_info(
            TableDescription(
                db_connection_id=db_connection.id,
                schema_name=SCHEMA,
                table_name=table,
                status=TableDescriptionStatus.NOT_SCANNED.value,
            )
        )
        created += 1
    print(f"→ Table descriptions: {created} created, {len(tables) - created} existed.")

    print(
        "\n✅ Done. db_connection_id = "
        f"{db_connection.id}\n"
        "Next: set OPENAI_API_KEY in .env, start the server, then run a full "
        "scan:\n"
        f'  curl -X POST localhost:8000/api/v1/table-descriptions/sync-schemas \\\n'
        f'    -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \\\n'
        f'    -d \'{{"ids": ["{db_connection.id}"]}}\''
    )


if __name__ == "__main__":
    main()
