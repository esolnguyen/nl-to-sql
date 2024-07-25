from abc import ABC, abstractmethod
from typing import List
from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.tools.base import BaseTool
from langchain_openai import OpenAIEmbeddings
from pydantic import Field
from models.db_description import TableDescription
from utils.sql_database import SQLDatabase


class AgentToolkit(ABC, BaseToolkit):
    db: SQLDatabase = Field(exclude=True)
    context: List[dict] | None = Field(exclude=True, default=None)
    few_shot_examples: List[dict] | None = Field(exclude=True, default=None)
    instructions: List[dict] | None = Field(exclude=True, default=None)
    db_scan: List[TableDescription] = Field(exclude=True)
    embedding: OpenAIEmbeddings = Field(exclude=True)
    is_multiple_schema: bool = False

    @property
    def dialect(self) -> str:
        return self.db.dialect

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def get_tools(self) -> List[BaseTool]:
        pass
