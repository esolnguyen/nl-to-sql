from abc import ABC, abstractmethod

from app.databases.sql_database import SQLDatabase
from app.models.db_description import ScannerRequest, TableDescription
from app.repositories.query_histories import QueryHistoryRepository
from app.repositories.table_descriptions import TableDescriptionRepository
from app.config import Component


class Scanner(Component, ABC):
    @abstractmethod
    def scan(
        self,
        db_engine: SQLDatabase,
        table_descriptions: list[TableDescription],
        repository: TableDescriptionRepository,
        query_history_repository: QueryHistoryRepository,
    ) -> None:
        pass

    @abstractmethod
    def synchronizing(
        self,
        scanner_request: ScannerRequest,
        repository: TableDescriptionRepository,
    ) -> list[TableDescription]:
        pass

    @abstractmethod
    def create_tables(
        self,
        tables: list[str],
        db_connection_id: str,
        schema: str,
        repository: TableDescriptionRepository,
        metadata: dict = None,
    ) -> None:
        pass

    @abstractmethod
    def refresh_tables(
        self,
        schemas_and_tables: dict[str, list],
        db_connection_id: str,
        repository: TableDescriptionRepository,
        metadata: dict = None,
    ) -> list[TableDescription]:
        pass
