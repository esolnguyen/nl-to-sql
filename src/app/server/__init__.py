from abc import ABC, abstractmethod

from app.config import Settings


class NlToSQLServer(ABC):
    @abstractmethod
    def __init__(self, settings: Settings):
        pass
