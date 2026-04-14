from pydantic import BaseModel, Field
from datetime import datetime


class Prompt(BaseModel):
    id: str | None = None
    text: str
    db_connection_id: str
    schemas: list[str] | None
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: dict | None


class PromptRequest(BaseModel):
    text: str
    db_connection_id: str
    schemas: list[str] | None
    metadata: dict | None
