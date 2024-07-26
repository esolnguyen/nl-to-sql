from abc import ABC, abstractmethod

from app.models.finetuning import Finetuning
from app.config import Component


class FinetuningModel(Component, ABC):
    def __init__(self, storage):
        self.storage = storage

    @abstractmethod
    def count_tokens(self, messages: dict) -> int:
        pass

    @abstractmethod
    def create_fintuning_dataset(self):
        pass

    @abstractmethod
    def create_fine_tuning_job(self):
        pass

    @abstractmethod
    def retrieve_finetuning_job(self) -> Finetuning:
        pass

    @abstractmethod
    def cancel_finetuning_job(self) -> Finetuning:
        pass
