import io
from abc import ABC, abstractmethod
from typing import List
from fastapi import BackgroundTasks
from src.app.api.types.requests import (
    NLGenerationRequest,
    NLGenerationsSQLGenerationRequest,
    PromptSQLGenerationNLGenerationRequest,
    PromptSQLGenerationRequest,
    StreamPromptSQLGenerationRequest,
    UpdateMetadataRequest,
)
from src.app.api.types.responses import (
    DatabaseConnectionResponse,
    ExampleSQLResponse,
    InstructionResponse,
    NLGenerationResponse,
    PromptResponse,
    SQLGenerationResponse,
    TableDescriptionResponse,
)
from src.app.models.db_conntection import (
    DatabaseConnection,
    DatabaseConnectionRequest,
)
from src.app.models.db_description import (
    RefreshTableDescriptionRequest,
    ScannerRequest,
    TableDescriptionRequest,
)
from models.example_sql import ExampleSQL, ExampleSQLRequest
from models.finetuning import (
    CancelFineTuningRequest,
    FineTuningRequest,
    Finetuning,
)
from models.instruction import InstructionRequest, UpdateInstruction
from models.prompt import PromptRequest
from models.query_history import QueryHistory
from models.sql_generation import SQLGenerationRequest
from config import Component


class API(Component, ABC):
    @abstractmethod
    def heartbeat(self) -> int:
        """Returns the current server time in nanoseconds to check if the server is alive"""
        pass

    @abstractmethod
    def scan_db(
        self, scanner_request: ScannerRequest, background_tasks: BackgroundTasks
    ) -> list[TableDescriptionResponse]:
        pass

    @abstractmethod
    def refresh_table_description(
        self, refresh_table_description: RefreshTableDescriptionRequest
    ) -> list[TableDescriptionResponse]:
        pass

    @abstractmethod
    def create_database_connection(
        self, database_connection_request: DatabaseConnectionRequest
    ) -> DatabaseConnectionResponse:
        pass

    @abstractmethod
    def list_database_connections(self) -> list[DatabaseConnection]:
        pass

    @abstractmethod
    def update_database_connection(
        self,
        db_connection_id: str,
        database_connection_request: DatabaseConnectionRequest,
    ) -> DatabaseConnection:
        pass

    @abstractmethod
    def update_table_description(
        self,
        table_description_id: str,
        table_description_request: TableDescriptionRequest,
    ) -> TableDescriptionResponse:
        pass

    @abstractmethod
    def list_table_descriptions(
        self, db_connection_id: str, table_name: str | None = None
    ) -> list[TableDescriptionResponse]:
        pass

    @abstractmethod
    def get_table_description(
        self, table_description_id: str
    ) -> TableDescriptionResponse:
        pass

    @abstractmethod
    def create_prompt(self, prompt_request: PromptRequest) -> PromptResponse:
        pass

    @abstractmethod
    def get_prompt(self, prompt_id) -> PromptResponse:
        pass

    @abstractmethod
    def update_prompt(
        self, prompt_id: str, update_metadata_request: UpdateMetadataRequest
    ) -> PromptResponse:
        pass

    @abstractmethod
    def get_prompts(self, db_connection_id: str | None = None) -> List[PromptResponse]:
        pass

    @abstractmethod
    def add_example_sqls(
        self, example_sqls: List[ExampleSQLRequest]
    ) -> List[ExampleSQLResponse]:
        pass

    @abstractmethod
    def execute_sql_query(self, sql_generation_id: str, max_rows: int = 100) -> list:
        pass

    @abstractmethod
    def export_csv_file(self, sql_generation_id: str) -> io.StringIO:
        pass

    @abstractmethod
    def get_query_history(self, db_connection_id: str) -> list[QueryHistory]:
        pass

    @abstractmethod
    def delete_example_sql(self, example_sql_id: str) -> dict:
        pass

    @abstractmethod
    def get_example_sqls(
        self, db_connection_id: str = None, page: int = 1, limit: int = 10
    ) -> List[ExampleSQL]:
        pass

    @abstractmethod
    def update_example_sql(
        self, example_sql_id: str, update_metadata_request: UpdateMetadataRequest
    ) -> ExampleSQL:
        pass

    @abstractmethod
    def add_instruction(
        self, instruction_request: InstructionRequest
    ) -> InstructionResponse:
        pass

    @abstractmethod
    def get_instructions(
        self, db_connection_id: str = None, page: int = 1, limit: int = 10
    ) -> List[InstructionResponse]:
        pass

    @abstractmethod
    def delete_instruction(self, instruction_id: str) -> dict:
        pass

    @abstractmethod
    def update_instruction(
        self,
        instruction_id: str,
        instruction_request: UpdateInstruction,
    ) -> InstructionResponse:
        pass

    @abstractmethod
    def create_finetuning_job(
        self, fine_tuning_request: FineTuningRequest, background_tasks: BackgroundTasks
    ) -> Finetuning:
        pass

    @abstractmethod
    def cancel_finetuning_job(
        self, cancel_fine_tuning_request: CancelFineTuningRequest
    ) -> Finetuning:
        pass

    @abstractmethod
    def get_finetunings(self, db_connection_id: str | None = None) -> list[Finetuning]:
        pass

    @abstractmethod
    def delete_finetuning_job(self, finetuning_job_id: str) -> dict:
        pass

    @abstractmethod
    def get_finetuning_job(self, finetuning_job_id: str) -> Finetuning:
        pass

    @abstractmethod
    def update_finetuning_job(
        self, finetuning_job_id: str, update_metadata_request: UpdateMetadataRequest
    ) -> Finetuning:
        pass

    @abstractmethod
    def create_sql_generation(
        self, prompt_id: str, sql_generation_request: SQLGenerationRequest
    ) -> SQLGenerationResponse:
        pass

    @abstractmethod
    def create_prompt_and_sql_generation(
        self, prompt_sql_generation_request: PromptSQLGenerationRequest
    ) -> SQLGenerationResponse:
        pass

    @abstractmethod
    def get_sql_generations(
        self, prompt_id: str | None = None
    ) -> list[SQLGenerationResponse]:
        pass

    @abstractmethod
    def get_sql_generation(self, sql_generation_id: str) -> SQLGenerationResponse:
        pass

    @abstractmethod
    def update_sql_generation(
        self, sql_generation_id: str, update_metadata_request: UpdateMetadataRequest
    ) -> SQLGenerationResponse:
        pass

    @abstractmethod
    def create_nl_generation(
        self, sql_generation_id: str, nl_generation_request: NLGenerationRequest
    ) -> NLGenerationResponse:
        pass

    @abstractmethod
    def create_sql_and_nl_generation(
        self,
        prompt_id: str,
        nl_generation_sql_generation_request: NLGenerationsSQLGenerationRequest,
    ) -> NLGenerationResponse:
        pass

    def create_prompt_sql_and_nl_generation(
        self, request: PromptSQLGenerationNLGenerationRequest
    ) -> NLGenerationResponse:
        pass

    @abstractmethod
    def get_nl_generations(
        self, sql_generation_id: str | None = None
    ) -> list[NLGenerationResponse]:
        pass

    @abstractmethod
    def get_nl_generation(self, nl_generation_id: str) -> NLGenerationResponse:
        pass

    @abstractmethod
    def update_nl_generation(
        self, nl_generation_id: str, update_metadata_request: UpdateMetadataRequest
    ) -> NLGenerationResponse:
        pass

    @abstractmethod
    async def stream_create_prompt_and_sql_generation(
        self,
        request: StreamPromptSQLGenerationRequest,
    ):
        pass
