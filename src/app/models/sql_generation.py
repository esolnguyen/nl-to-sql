from pydantic import BaseModel, Field, validator
from app.models.llm import LLMConfig
from datetime import datetime
from enum import Enum
from sql_metadata import Parser


class SQLGenerationStatus(Enum):
    NONE = "NONE"
    VALID = "VALID"
    INVALID = "INVALID"


class IntermediateStep(BaseModel):
    thought: str
    action: str
    action_input: str
    observation: str


class SQLGeneration(BaseModel):
    id: str | None = None
    prompt_id: str
    finetuning_id: str | None
    low_latency_mode: bool = False
    llm_config: LLMConfig | None
    evaluate: bool = False
    intermediate_steps: list[IntermediateStep] | None
    sql: str | None
    status: str = "INVALID"
    completed_at: datetime | None
    tokens_used: int | None
    confidence_score: float | None
    error: str | None
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: dict | None


class SQLGenerationRequest(BaseModel):
    finetuning_id: str | None
    low_latency_mode: bool = False
    llm_config: LLMConfig | None
    evaluate: bool = False
    sql: str | None
    metadata: dict | None

    @validator("sql")
    def validate_model_name(cls, v: str | None):
        try:
            Parser(v).tables
        except Exception as e:
            raise ValueError(f"SQL {v} is malformed. Please check the syntax.") from e
        return v
