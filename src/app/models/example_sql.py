from app.models.db_conntection import DBConnectionValidation
from pydantic import BaseModel, Field
from datetime import datetime


class ExampleSQLRequest(DBConnectionValidation):
    prompt_text: str = Field(None, min_length=3)
    sql: str = Field(None, min_length=3)
    metadata: dict | None


class ExampleSQL(BaseModel):
    id: str | None = None
    prompt_text: str
    sql: str
    db_connection_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: dict | None
