from models.db_conntection import DBConnectionValidation
from pydantic import BaseModel, Field
from datetime import datetime


class UpdateInstruction(BaseModel):
    instruction: str
    metadata: dict | None


class InstructionRequest(DBConnectionValidation):
    instruction: str = Field(None, min_length=3)
    metadata: dict | None


class Instruction(BaseModel):
    id: str | None = None
    instruction: str
    db_connection_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: dict | None
