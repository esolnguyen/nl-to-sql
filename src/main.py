from config import Settings
from server.fastapi import FastAPI


settings = Settings()
server = FastAPI(settings)
app = server.app()
