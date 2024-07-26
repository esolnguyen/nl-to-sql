from abc import ABC, abstractmethod

from sqlalchemy.sql.schema import Column

from src.app.databases.sql_database import SQLDatabase
from src.app.models.query_history import QueryHistory


class AbstractScanner(ABC):
    @abstractmethod
    def cardinality_values(self, column: Column, db_engine: SQLDatabase) -> list | None:
        pass

    @abstractmethod
    def get_logs(
        self, table: str, db_engine: SQLDatabase, db_connection_id: str
    ) -> list[QueryHistory]:
        pass
