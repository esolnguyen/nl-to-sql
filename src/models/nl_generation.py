from pydantic import BaseModel, Field
from models.llm import LLMConfig
from datetime import datetime


class NLGeneration(BaseModel):
    id: str | None = None
    sql_generation_id: str
    llm_config: LLMConfig | None
    text: str | None
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: dict | None
