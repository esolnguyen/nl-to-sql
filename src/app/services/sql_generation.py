import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime
from queue import Queue
import pandas as pd
from src.app.models.llm import LLMConfig
from src.app.models.sql_generation import SQLGeneration, SQLGenerationRequest
from src.app.repositories.prompts import PromptRepository
from src.app.repositories.sql_generations import (
    SQLGenerationNotFoundError,
    SQLGenerationRepository,
)
from src.app.models.prompt import Prompt
from src.app.llm_agents.finetuning_agent import FinetuningAgent
from src.app.llm_agents.sql_generation_agent import SQLGenerationAgent
from src.app.utils.custom_error import (
    EmptySQLGenerationError,
    PromptNotFoundError,
    SQLGenerationError,
)
from src.app.databases.sql_database import SQLDatabase, create_sql_query_status
from src.config import System
from src.app.repositories.db_connections import DatabaseConnectionRepository


class SQLGenerationService:
    def __init__(self, system: System, storage):
        self.system = system
        self.storage = storage
        self.sql_generation_repository = SQLGenerationRepository(storage)
        self.prompt_repository = PromptRepository(storage)

    def create_sql_generator(
        self, prompt_id: str, sql_generation_request: SQLGenerationRequest
    ) -> SQLGeneration:
        initial_sql_generation = self._create_init_sql_generation(
            prompt_id, sql_generation_request.llm_config
        )
        self.sql_generation_repository.insert(initial_sql_generation)
        prompt = self._get_promp_by_id(prompt_id)
        db_connection = self._get_db_connection_by_id(prompt.db_connection_id)
        database = SQLDatabase.get_sql_engine(db_connection, True)

        if sql_generation_request.sql is not None:
            sql_generation = SQLGeneration(
                prompt_id=prompt_id,
                sql=sql_generation_request.sql,
                tokens_used=0,
            )
            try:
                sql_generation = create_sql_query_status(
                    db=database, query=sql_generation.sql, sql_generation=sql_generation
                )
            except Exception as e:
                self._update_error(initial_sql_generation, str(e))
                raise SQLGenerationError(str(e), initial_sql_generation.id) from e
        else:
            sql_generator = self._create_generator_agent(
                prompt_id, sql_generation_request
            )
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        self._generate_response_with_timeout,
                        sql_generator,
                        prompt,
                        db_connection,
                    )
                    try:
                        sql_generation = future.result(
                            timeout=int(os.environ.get("DH_ENGINE_TIMEOUT", 150))
                        )
                    except TimeoutError as e:
                        self._update_error(
                            initial_sql_generation, "SQL generation request timed out"
                        )
                        raise SQLGenerationError(
                            "SQL generation request timed out",
                            initial_sql_generation.id,
                        ) from e
            except Exception as e:
                self._update_error(initial_sql_generation, str(e))
                raise SQLGenerationError(str(e), initial_sql_generation.id) from e
        return self._update_the_initial_sql_generation(
            initial_sql_generation, sql_generation
        )

    def start_streaming(
        self, prompt_id: str, sql_generation_request: SQLGenerationRequest, queue: Queue
    ):
        initial_sql_generation = self._create_init_sql_generation(
            prompt_id, sql_generation_request.llm_config
        )
        self.sql_generation_repository.insert(initial_sql_generation)
        prompt = self._get_promp_by_id(prompt_id)
        db_connection = self._get_db_connection_by_id(prompt.db_connection_id)
        sql_generator = self._create_generator_agent(prompt_id, sql_generation_request)
        try:
            sql_generator.stream_response(
                user_prompt=prompt,
                database_connection=db_connection,
                response=initial_sql_generation,
                queue=queue,
            )
        except Exception as e:
            self._update_error(initial_sql_generation, str(e))
            raise SQLGenerationError(str(e), initial_sql_generation.id) from e

    def get_sql_generation(self, query) -> list[SQLGeneration]:
        return self.sql_generation_repository.find_by(query)

    def execute_sql(
        self, sql_generation_id: str, max_rows: int = 100
    ) -> tuple[str, dict]:
        sql_generation = self.sql_generation_repository.find_by_id(sql_generation_id)
        if not sql_generation:
            raise SQLGenerationNotFoundError(
                f"SQL Generation {sql_generation_id} not found"
            )
        prompt_repository = PromptRepository(self.storage)
        prompt = prompt_repository.find_by_id(sql_generation.prompt_id)
        db_connection_repository = DatabaseConnectionRepository(self.storage)
        db_connection = db_connection_repository.find_by_id(prompt.db_connection_id)
        database = SQLDatabase.get_sql_engine(db_connection, True)
        return database.run_sql(sql_generation.sql, max_rows)

    def update_metadata(self, sql_generation_id, metadata_request) -> SQLGeneration:
        sql_generation = self.sql_generation_repository.find_by_id(sql_generation_id)
        if not sql_generation:
            raise SQLGenerationNotFoundError(
                f"Sql generation {sql_generation_id} not found"
            )
        sql_generation.metadata = metadata_request.metadata
        return self.sql_generation_repository.update(sql_generation)

    def create_dataframe(self, sql_generation_id):
        sql_generation = self.sql_generation_repository.find_by_id(sql_generation_id)
        if not sql_generation:
            raise SQLGenerationNotFoundError(
                f"Sql generation {sql_generation_id} not found"
            )
        prompt_repository = PromptRepository(self.storage)
        prompt = prompt_repository.find_by_id(sql_generation.prompt_id)
        db_connection_repository = DatabaseConnectionRepository(self.storage)
        db_connection = db_connection_repository.find_by_id(prompt.db_connection_id)
        database = SQLDatabase.get_sql_engine(db_connection)
        results = database.run_sql(sql_generation.sql)
        if results is None:
            raise EmptySQLGenerationError(
                f"Sql generation {sql_generation_id} is empty"
            )
        data = results[1]["result"]
        return pd.DataFrame(data)

    def _update_error(self, sql_generation: SQLGeneration, error: str) -> SQLGeneration:
        sql_generation.error = error
        return self.sql_generation_repository.update(sql_generation)

    def _generate_response_with_timeout(
        self, sql_generator, user_prompt, db_connection, metadata=None
    ):
        return sql_generator.generate_response(
            user_prompt=user_prompt,
            database_connection=db_connection,
            metadata=metadata,
        )

    def _update_the_initial_sql_generation(
        self, initial_sql_generation: SQLGeneration, sql_generation: SQLGeneration
    ):
        initial_sql_generation.sql = sql_generation.sql
        initial_sql_generation.tokens_used = sql_generation.tokens_used
        initial_sql_generation.completed_at = datetime.now()
        initial_sql_generation.status = sql_generation.status
        initial_sql_generation.error = sql_generation.error
        initial_sql_generation.intermediate_steps = sql_generation.intermediate_steps
        return self.sql_generation_repository.update(initial_sql_generation)

    def _get_promp_by_id(
        self, prompt_id: str, initial_sql_generation: SQLGeneration
    ) -> Prompt:
        prompt = self.prompt_repository.find_by_id(prompt_id)
        if not prompt:
            self._update_error(initial_sql_generation, f"Prompt {prompt_id} not found")
            raise PromptNotFoundError(
                f"Prompt {prompt_id} not found", initial_sql_generation.id
            )
        return prompt

    def _get_db_connection_by_id(self, db_connection_id: str):
        db_connection_repository = DatabaseConnectionRepository(self.storage)
        return db_connection_repository.find_by_id(db_connection_id)

    def _create_init_sql_generation(
        self, prompt_id: str, llm_config: LLMConfig
    ) -> SQLGeneration:
        return SQLGeneration(
            prompt_id=prompt_id,
            created_at=datetime.now(),
            llm_config=llm_config if llm_config else LLMConfig(),
        )

    def _create_generator_agent(
        self,
        sql_generation_request: SQLGenerationRequest,
        initial_sql_generation: SQLGeneration,
    ) -> SQLGeneration:
        if (
            sql_generation_request.finetuning_id is None
            or sql_generation_request.finetuning_id == ""
        ):
            if sql_generation_request.low_latency_mode:
                raise SQLGenerationError(
                    "Low latency mode is not supported for our old agent with no finetuning. Please specify a finetuning id.",
                    initial_sql_generation.id,
                )
            sql_generator = SQLGenerationAgent(
                self.system,
                (
                    sql_generation_request.llm_config
                    if sql_generation_request.llm_config
                    else LLMConfig()
                ),
            )
        else:
            sql_generator = FinetuningAgent(
                self.system,
                (
                    sql_generation_request.llm_config
                    if sql_generation_request.llm_config
                    else LLMConfig()
                ),
            )
            sql_generator.finetuning_id = sql_generation_request.finetuning_id
            sql_generator.use_fintuned_model_only = (
                sql_generation_request.low_latency_mode
            )
            initial_sql_generation.finetuning_id = sql_generation_request.finetuning_id
            initial_sql_generation.low_latency_mode = (
                sql_generation_request.low_latency_mode
            )
        return sql_generator
