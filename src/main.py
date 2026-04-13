from app.config import Settings
from app.server.fastapi import FastAPI


settings = Settings()
server = FastAPI(settings)
app = server.app()
