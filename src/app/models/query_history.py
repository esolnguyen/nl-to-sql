from pydantic import BaseModel


class QueryHistory(BaseModel):
    id: str | None
    db_connection_id: str
    table_name: str
    query: str
    user: str
    occurrences: int = 0
