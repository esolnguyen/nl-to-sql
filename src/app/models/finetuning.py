from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum
from app.constants.model_contexts import OPENAI_FINETUNING_MODELS_WINDOW_SIZES


class BaseLLM(BaseModel):
    model_provider: str | None = None
    model_name: str | None = None
    model_parameters: dict[str, str] | None = None

    @validator("model_name")
    def validate_model_name(cls, v: str | None):
        if v and v not in OPENAI_FINETUNING_MODELS_WINDOW_SIZES:
            raise ValueError(f"Model {v} not supported")
        return v


class Finetuning(BaseModel):
    id: str | None = None
    alias: str | None = None
    db_connection_id: str | None = None
    schemas: list[str] | None
    status: str = "QUEUED"
    error: str | None = None
    base_llm: BaseLLM | None = None
    finetuning_file_id: str | None = None
    finetuning_job_id: str | None = None
    model_id: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    example_sqls: list[str] | None = None
    metadata: dict | None


class FineTuningStatus(Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    VALIDATING_FILES = "VALIDATING_FILES"


class FineTuningRequest(BaseModel):
    db_connection_id: str
    schemas: list[str] | None
    alias: str | None = None
    base_llm: BaseLLM | None = None
    example_sqls: list[str] | None = None
    metadata: dict | None


class CancelFineTuningRequest(BaseModel):
    finetuning_id: str
    metadata: dict | None
