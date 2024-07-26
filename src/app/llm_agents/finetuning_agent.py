import datetime
import logging
import os
from queue import Queue
from threading import Thread
from typing import Any, Dict, List
from langchain.agents.agent import AgentExecutor
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain_community.callbacks import get_openai_callback
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from overrides import override
from pydantic import Field
from app.constants.finetuning_prompts import (
    FINETUNING_AGENT_PREFIX,
    FINETUNING_AGENT_PREFIX_FINETUNING_ONLY,
    FINETUNING_AGENT_SUFFIX,
)
from app.constants.model_contexts import EMBEDDING_MODEL
from app.constants.sql_generation_prompts import (
    ERROR_PARSING_MESSAGE,
    FORMAT_INSTRUCTIONS,
)
from app.databases.mongodb import NlToSQLDatabase
from app.databases.sql_database import SQLDatabase, SQLInjectionError
from app.llm_agents import (
    EngineTimeOutORItemLimitError,
    SQLGenerator,
    replace_unprocessable_characters,
)
from app.llm_agents.toolkit.finetuning_agent_toolkit import (
    FineTuningAgentToolkit,
)
from app.models.db_conntection import DatabaseConnection
from app.models.db_description import TableDescriptionStatus
from app.models.finetuning import FineTuningStatus
from app.models.prompt import Prompt
from app.models.sql_generation import SQLGeneration
from app.repositories.finetunings import FinetuningsRepository
from app.repositories.sql_generations import SQLGenerationRepository
from app.repositories.table_descriptions import TableDescriptionRepository
from app.services.context_store import ContextStore
from app.services.finetuning.openai_finetuning import OpenAIFineTuning
from app.utils.custom_error import FinetuningNotAvailableError

logger = logging.getLogger(__name__)


class FinetuningAgent(SQLGenerator):
    llm: Any = None
    finetuning_id: str = Field(exclude=True)
    use_fintuned_model_only: bool = Field(exclude=True, default=False)

    def create_sql_agent(
        self,
        toolkit: FineTuningAgentToolkit,
        callback_manager: BaseCallbackManager | None = None,
        prefix: str = FINETUNING_AGENT_PREFIX,
        suffix: str = FINETUNING_AGENT_SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        input_variables: List[str] | None = None,
        max_iterations: int | None = int(os.getenv("AGENT_MAX_ITERATIONS", "12")),
        max_execution_time: float | None = None,
        early_stopping_method: str = "generate",
        verbose: bool = False,
        agent_executor_kwargs: Dict[str, Any] | None = None,
        **kwargs: Dict[str, Any],
    ) -> AgentExecutor:
        tools = toolkit.get_tools()
        admin_instructions = ""
        if toolkit.instructions:
            for index, instruction in enumerate(toolkit.instructions):
                admin_instructions += f"{index+1}) {instruction['instruction']}\n"
        if self.use_fintuned_model_only:
            prefix = FINETUNING_AGENT_PREFIX_FINETUNING_ONLY
        prefix = prefix.format(
            dialect=toolkit.dialect, admin_instructions=admin_instructions
        )
        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix,
            format_instructions=format_instructions,
            input_variables=input_variables,
        )
        llm_chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names, **kwargs)
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            callback_manager=callback_manager,
            verbose=verbose,
            max_iterations=max_iterations,
            max_execution_time=max_execution_time,
            early_stopping_method=early_stopping_method,
            **(agent_executor_kwargs or {}),
        )

    @override
    def generate_response(
        self,
        user_prompt: Prompt,
        database_connection: DatabaseConnection,
        context: List[dict] = None,
        metadata: dict = None,
    ) -> SQLGeneration:
        context_store = self.system.instance(ContextStore)
        storage = self.system.instance(NlToSQLDatabase)
        response = SQLGeneration(
            prompt_id=user_prompt.id,
            created_at=datetime.datetime.now(),
            llm_config=self.llm_config,
            finetuning_id=self.finetuning_id,
        )
        self.llm = self.model.get_model(
            database_connection=database_connection,
            temperature=0,
            model_name=self.llm_config.llm_name,
            api_base=self.llm_config.api_base,
        )
        repository = TableDescriptionRepository(storage)
        db_scan = repository.get_all_tables_by_db(
            {
                "db_connection_id": str(database_connection.id),
                "status": TableDescriptionStatus.SCANNED.value,
            }
        )
        if not db_scan:
            raise ValueError("No scanned tables found for database")
        db_scan = SQLGenerator.filter_tables_by_schema(
            db_scan=db_scan, prompt=user_prompt
        )
        few_shot_examples, instructions = context_store.retrieve_context_for_question(
            user_prompt, number_of_samples=5
        )
        finetunings_repository = FinetuningsRepository(storage)
        finetuning = finetunings_repository.find_by_id(self.finetuning_id)
        openai_fine_tuning = OpenAIFineTuning(self.system, storage, finetuning)
        finetuning = openai_fine_tuning.retrieve_finetuning_job()
        if finetuning.status != FineTuningStatus.SUCCEEDED.value:
            raise FinetuningNotAvailableError(
                f"Finetuning({self.finetuning_id}) has the status {finetuning.status}."
                f"Finetuning should have the status {FineTuningStatus.SUCCEEDED.value} to generate SQL queries."
            )
        self.database = SQLDatabase.get_sql_engine(database_connection)
        if self.system.settings["azure_api_key"] is not None:
            embedding = AzureOpenAIEmbeddings(
                openai_api_key=database_connection.decrypt_api_key(),
                model=EMBEDDING_MODEL,
            )
        else:
            embedding = OpenAIEmbeddings(
                openai_api_key=database_connection.decrypt_api_key(),
                model=EMBEDDING_MODEL,
            )
        toolkit = FineTuningAgentToolkit(
            db=self.database,
            instructions=instructions,
            few_shot_examples=few_shot_examples,
            db_scan=db_scan,
            api_key=database_connection.decrypt_api_key(),
            finetuning_model_id=finetuning.model_id,
            use_finetuned_model_only=self.use_fintuned_model_only,
            model_name=finetuning.base_llm.model_name,
            openai_fine_tuning=openai_fine_tuning,
            embedding=embedding,
        )
        agent_executor = self.create_sql_agent(
            toolkit=toolkit,
            verbose=True,
            max_execution_time=int(os.environ.get("DH_ENGINE_TIMEOUT", 150)),
        )
        agent_executor.return_intermediate_steps = True
        agent_executor.handle_parsing_errors = ERROR_PARSING_MESSAGE
        with get_openai_callback() as cb:
            try:
                result = agent_executor.invoke(
                    {"input": user_prompt.text}, {"metadata": metadata}
                )
                result = self.check_for_time_out_or_tool_limit(result)
            except SQLInjectionError as e:
                raise SQLInjectionError(e) from e
            except EngineTimeOutORItemLimitError as e:
                raise EngineTimeOutORItemLimitError(e) from e
            except Exception as e:
                return SQLGeneration(
                    prompt_id=user_prompt.id,
                    tokens_used=cb.total_tokens,
                    finetuning_id=self.finetuning_id,
                    completed_at=datetime.datetime.now(),
                    sql="",
                    status="INVALID",
                    error=str(e),
                )
        sql_query = ""
        if "```sql" in result["output"]:
            sql_query = self.remove_markdown(result["output"])
        else:
            sql_query = self.extract_query_from_intermediate_steps(
                result["intermediate_steps"]
            )
        logger.info(f"cost: {str(cb.total_cost)} tokens: {str(cb.total_tokens)}")
        response.sql = replace_unprocessable_characters(sql_query)
        response.tokens_used = cb.total_tokens
        response.completed_at = datetime.datetime.now()
        response.intermediate_steps = self.construct_intermediate_steps(
            result["intermediate_steps"], FINETUNING_AGENT_SUFFIX
        )
        return self.create_sql_query_status(
            self.database,
            response.sql,
            response,
        )

    @override
    def stream_response(
        self,
        user_prompt: Prompt,
        database_connection: DatabaseConnection,
        response: SQLGeneration,
        queue: Queue,
        metadata: dict = None,
    ):
        context_store = self.system.instance(ContextStore)
        storage = self.system.instance(NlToSQLDatabase)
        sql_generation_repository = SQLGenerationRepository(storage)
        self.llm = self.model.get_model(
            database_connection=database_connection,
            temperature=0,
            model_name=self.llm_config.llm_name,
            api_base=self.llm_config.api_base,
            streaming=True,
        )
        repository = TableDescriptionRepository(storage)
        db_scan = repository.get_all_tables_by_db(
            {
                "db_connection_id": str(database_connection.id),
                "status": TableDescriptionStatus.SCANNED.value,
            }
        )
        if not db_scan:
            raise ValueError("No scanned tables found for database")
        db_scan = SQLGenerator.filter_tables_by_schema(
            db_scan=db_scan, prompt=user_prompt
        )
        _, instructions = context_store.retrieve_context_for_question(
            user_prompt, number_of_samples=1
        )
        finetunings_repository = FinetuningsRepository(storage)
        finetuning = finetunings_repository.find_by_id(self.finetuning_id)
        openai_fine_tuning = OpenAIFineTuning(self.system, storage, finetuning)
        finetuning = openai_fine_tuning.retrieve_finetuning_job()
        if finetuning.status != FineTuningStatus.SUCCEEDED.value:
            raise FinetuningNotAvailableError(
                f"Finetuning({self.finetuning_id}) has the status {finetuning.status}."
                f"Finetuning should have the status {FineTuningStatus.SUCCEEDED.value} to generate SQL queries."
            )
        self.database = SQLDatabase.get_sql_engine(database_connection)
        if self.system.settings["azure_api_key"] is not None:
            embedding = AzureOpenAIEmbeddings(
                openai_api_key=database_connection.decrypt_api_key(),
                model=EMBEDDING_MODEL,
            )
        else:
            embedding = OpenAIEmbeddings(
                openai_api_key=database_connection.decrypt_api_key(),
                model=EMBEDDING_MODEL,
            )
        toolkit = FineTuningAgentToolkit(
            db=self.database,
            instructions=instructions,
            db_scan=db_scan,
            api_key=database_connection.decrypt_api_key(),
            finetuning_model_id=finetuning.model_id,
            use_finetuned_model_only=self.use_fintuned_model_only,
            model_name=finetuning.base_llm.model_name,
            openai_fine_tuning=openai_fine_tuning,
            embedding=embedding,
        )
        agent_executor = self.create_sql_agent(
            toolkit=toolkit,
            verbose=True,
            max_execution_time=int(os.environ.get("DH_ENGINE_TIMEOUT", 150)),
        )
        agent_executor.return_intermediate_steps = True
        agent_executor.handle_parsing_errors = ERROR_PARSING_MESSAGE
        thread = Thread(
            target=self.stream_agent_steps,
            args=(
                user_prompt.text,
                agent_executor,
                response,
                sql_generation_repository,
                queue,
                metadata,
            ),
        )
        thread.start()
