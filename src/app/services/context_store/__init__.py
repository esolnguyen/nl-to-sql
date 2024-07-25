import os
from abc import ABC, abstractmethod
from typing import List, Tuple
from databases.mongodb import NlToSQLDatabase
from databases.vector import VectorDatabase
from src.app.models.example_sql import ExampleSQL, ExampleSQLRequest
from src.app.models.prompt import Prompt
from config import Component, System


class ContextStore(Component, ABC):
    DocStore: NlToSQLDatabase
    VectorStore: VectorDatabase
    doc_store_collection = "table_meta_data"

    @abstractmethod
    def __init__(self, system: System):
        self.system = system
        self.db = self.system.instance(NlToSQLDatabase)
        self.example_sql_collection = os.environ.get(
            "EXAMPLE_SQL_COLLECTION", "ai-stage"
        )
        self.vector_store = self.system.instance(VectorDatabase)

    @abstractmethod
    def retrieve_context_for_question(
        self, prompt: Prompt, number_of_samples: int = 3
    ) -> Tuple[List[dict] | None, List[dict] | None]:
        pass

    @abstractmethod
    def add_example_sqls(
        self, example_sqls: List[ExampleSQLRequest]
    ) -> List[ExampleSQL]:
        pass

    @abstractmethod
    def remove_example_sqls(self, ids: List) -> bool:
        pass
