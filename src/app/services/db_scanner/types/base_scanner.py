import sqlalchemy
from overrides import override
from sqlalchemy.sql import func
from sqlalchemy.sql.schema import Column

from src.app.databases.sql_database import SQLDatabase
from src.app.models.query_history import QueryHistory
from src.app.services.db_scanner import AbstractScanner

MIN_CATEGORY_VALUE = 1
MAX_CATEGORY_VALUE = 100


class BaseScanner(AbstractScanner):
    @override
    def cardinality_values(self, column: Column, db_engine: SQLDatabase) -> list | None:
        cardinality_query = sqlalchemy.select([func.distinct(column)]).limit(101)
        cardinality = db_engine.engine.execute(cardinality_query).fetchall()

        if MAX_CATEGORY_VALUE > len(cardinality) > MIN_CATEGORY_VALUE:
            return [str(category[0]) for category in cardinality]
        return None

    @override
    def get_logs(
        self, table: str, db_engine: SQLDatabase, db_connection_id: str
    ) -> list[QueryHistory]:
        return []
