from app.config import Settings
from src.app.server.fastapi import FastAPI


settings = Settings()
server = FastAPI(settings)
app = server.app()
