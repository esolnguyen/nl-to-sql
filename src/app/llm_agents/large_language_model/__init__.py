from abc import ABC, abstractmethod
from typing import Any
from app.config import Component, System
from app.models.db_conntection import DatabaseConnection


class LargeLanguageModel(Component, ABC):
    model: Any

    @abstractmethod
    def __init__(self, system: System):
        self.system = system

    @abstractmethod
    def get_model(
        self,
        database_connection: DatabaseConnection,
        model_family="openai",
        model_name="gpt-4-turbo-preview",
        api_base: str | None = None,
        **kwargs: Any
    ) -> Any:
        pass
