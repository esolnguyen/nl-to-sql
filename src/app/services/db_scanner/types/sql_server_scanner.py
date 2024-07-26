from overrides import override
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.schema import Column

from app.databases.sql_database import SQLDatabase
from app.models.query_history import QueryHistory
from app.services.db_scanner import AbstractScanner

MIN_CATEGORY_VALUE = 1
MAX_CATEGORY_VALUE = 100
MAX_LOGS = 5_000


class SqlServerScanner(AbstractScanner):
    @override
    def cardinality_values(self, column: Column, db_engine: SQLDatabase) -> list | None:
        try:
            count_query = f"SELECT APPROX_COUNT_DISTINCT({column.name}) FROM {column.table.name}"
            rs = db_engine.engine.execute(count_query).fetchall()
        except SQLAlchemyError:
            return None

        if (
            len(rs) > 0
            and len(rs[0]) > 0
            and MIN_CATEGORY_VALUE < rs[0][0] <= MAX_CATEGORY_VALUE
        ):
            cardinality_query = f"SELECT TOP 101 {column.name} FROM (SELECT DISTINCT {column.name} FROM [{column.table.name}]) AS subquery;"
            cardinality = db_engine.engine.execute(
                cardinality_query).fetchall()
            return [str(category[0]) for category in cardinality]

        return None

    @override
    def get_logs(
        self, table: str, db_engine: SQLDatabase, db_connection_id: str
    ) -> list[QueryHistory]:
        return []
