from abc import ABC, abstractmethod

from src.config import Settings


class NlToSQLServer(ABC):
    @abstractmethod
    def __init__(self, settings: Settings):
        pass
