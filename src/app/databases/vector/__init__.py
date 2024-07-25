from abc import ABC, abstractmethod
from typing import Any, List
from src.app.models.example_sql import ExampleSQL
from config import Component, System


class VectorDatabase(Component, ABC):
    collections: List[str]

    @abstractmethod
    def __init__(self, system: System):
        self.system = system

    @abstractmethod
    def query(
        self,
        query_texts: List[str],
        db_connection_id: str,
        collection: str,
        num_results: int,
    ) -> list:
        pass

    @abstractmethod
    def create_collection(self, collection: str):
        pass

    @abstractmethod
    def add_records(self, example_sqls: List[ExampleSQL], collection: str):
        pass

    @abstractmethod
    def add_record(
        self,
        documents: str,
        db_connection_id: str,
        collection: str,
        metadata: Any,
        ids: List = None,
    ):
        pass

    @abstractmethod
    def delete_record(self, collection: str, id: str):
        pass

    @abstractmethod
    def delete_collection(self, collection: str):
        pass
