from abc import ABC, abstractmethod

from config import Settings


class NlToSQLServer(ABC):
    @abstractmethod
    def __init__(self, settings: Settings):
        pass
