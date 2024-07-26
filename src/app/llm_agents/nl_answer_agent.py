from datetime import date, datetime
from decimal import Decimal

from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from sqlalchemy import text

from app.constants.sql_generation_prompts import HUMAN_TEMPLATE
from app.databases.sql_database import SQLDatabase, SQLInjectionError
from app.llm_agents.large_language_model.chat_model import ChatModel
from app.models.llm import LLMConfig
from app.models.nl_generation import NLGeneration
from app.models.sql_generation import SQLGeneration
from app.repositories.db_connections import DatabaseConnectionRepository
from app.repositories.prompts import PromptRepository


class GeneratesNlAnswer:
    def __init__(self, system, storage, llm_config: LLMConfig):
        self.system = system
        self.storage = storage
        self.llm_config = llm_config
        self.model = ChatModel(self.system)

    def execute(
        self,
        sql_generation: SQLGeneration,
        top_k: int = 100,
    ) -> NLGeneration:
        prompt_repository = PromptRepository(self.storage)
        prompt = prompt_repository.find_by_id(sql_generation.prompt_id)

        db_connection_repository = DatabaseConnectionRepository(self.storage)
        database_connection = db_connection_repository.find_by_id(
            prompt.db_connection_id
        )
        self.llm = self.model.get_model(
            database_connection=database_connection,
            temperature=0,
            model_name=self.llm_config.llm_name,
            api_base=self.llm_config.api_base,
        )
        database = SQLDatabase.get_sql_engine(database_connection, True)

        if sql_generation.status == "INVALID":
            return NLGeneration(
                sql_generation_id=sql_generation.id,
                text="I don't know, the SQL query is invalid.",
                created_at=datetime.now(),
            )

        try:
            query = database.parser_to_filter_commands(sql_generation.sql)
            with database._engine.connect() as connection:
                execution = connection.execute(text(query))
                result = execution.fetchmany(top_k)
            rows = []
            for row in result:
                modified_row = {}
                for key, value in zip(row.keys(), row, strict=True):
                    if type(value) in [
                        date,
                        datetime,
                    ]:
                        modified_row[key] = str(value)
                    elif (
                        type(value) is Decimal
                    ):
                        modified_row[key] = float(value)
                    else:
                        modified_row[key] = value
                rows.append(modified_row)

        except SQLInjectionError as e:
            raise SQLInjectionError(
                "Sensitive SQL keyword detected in the query."
            ) from e

        human_message_prompt = HumanMessagePromptTemplate.from_template(
            HUMAN_TEMPLATE)
        chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
        chain = LLMChain(llm=self.llm, prompt=chat_prompt)
        nl_resp = chain.invoke(
            {
                "prompt": prompt.text,
                "sql_query": sql_generation.sql,
                "sql_query_result": "\n".join([str(row) for row in rows]),
            }
        )
        return NLGeneration(
            sql_generation_id=sql_generation.id,
            llm_config=self.llm_config,
            text=nl_resp["text"],
            created_at=datetime.now(),
        )
