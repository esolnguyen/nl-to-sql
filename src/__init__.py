from src.config import Settings, System
from src.app.api import API


__settings = Settings()
__version__ = "0.0.1"


def client(settings: Settings = __settings) -> API:
    system = System(settings)
    api = system.instance(API)
    system.start()
    return api
