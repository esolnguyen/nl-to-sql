import datetime
import logging
import os
from typing import List, Type
from langchain_openai import OpenAIEmbeddings
import numpy as np
from openai import OpenAI
import pandas as pd
from pydantic import Field, BaseModel
from sql_metadata import Parser
from app.llm_agents.toolkit import AgentToolkit
from overrides import override
from langchain.tools.base import BaseTool
from app.models.db_description import TableDescription
from app.services.finetuning.openai_finetuning import OpenAIFineTuning
from app.constants.finetuning_prompts import FINETUNING_SYSTEM_INFORMATION
from app.constants.model_contexts import OPENAI_FINETUNING_MODELS_WINDOW_SIZES
from app.constants.sql import TOP_K, TOP_TABLES
from app.llm_agents import replace_unprocessable_characters
from app.utils.timeout import run_with_timeout
from app.utils.custom_error import catch_exceptions
from app.databases.sql_database import SQLDatabase
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

logger = logging.getLogger(__name__)


class SQLInput(BaseModel):
    sql_query: str = Field()


class QuestionInput(BaseModel):
    question: str = Field()


class FineTuningAgentToolkit(AgentToolkit):
    openai_fine_tuning: OpenAIFineTuning = Field(exclude=True)
    finetuning_model_id: str = Field(exclude=True)
    model_name: str = Field(exclude=True)
    api_key: str = Field(exclude=True)
    use_finetuned_model_only: bool = Field(exclude=True, default=None)

    @override
    def get_tools(self) -> List[BaseTool]:
        tools = []
        if not self.use_finetuned_model_only:
            tools.append(SystemTime(db=self.db))
            tools.append(SchemaSQLDatabaseTool(db=self.db, db_scan=self.db_scan))
            tools.append(
                TablesSQLDatabaseTool(
                    db=self.db,
                    db_scan=self.db_scan,
                    embedding=self.embedding,
                    few_shot_examples=self.few_shot_examples,
                )
            )
        tools.append(QuerySQLDataBaseTool(db=self.db))
        tools.append(
            GenerateSQL(
                db=self.db,
                db_scan=self.db_scan,
                api_key=self.api_key,
                finetuning_model_id=self.finetuning_model_id,
                model_name=self.model_name,
                openai_fine_tuning=self.openai_fine_tuning,
                embedding=self.embedding,
            )
        )
        return tools


class BaseSQLDatabaseTool(BaseModel):
    db: SQLDatabase = Field(exclude=True)

    class Config(BaseTool.Config):
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True
        extra = "allow"


class SystemTime(BaseSQLDatabaseTool, BaseTool):
    name = "SystemTime"
    description = """
    Input: None.
    Output: Current date and time.
    Use this tool to replace current_time and current_date in SQL queries with the actual current time and date.
    """

    @catch_exceptions()
    def _run(
        self,
        tool_input: str = "",
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        current_datetime = datetime.datetime.now()
        return f"Current Date and Time: {str(current_datetime)}"

    async def _arun(
        self,
        tool_input: str = "",
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        raise NotImplementedError("SystemTime tool does not support async")


class TablesSQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):
    name = "DbTablesWithRelevanceScores"
    description = """
    Input: Given question.
    Output: Comma-separated list of tables with their relevance scores, indicating their relevance to the question.
    Use this tool to identify the relevant tables for the given question.
    """
    db_scan: List[TableDescription]
    embedding: OpenAIEmbeddings
    few_shot_examples: List[dict] | None = Field(exclude=True, default=None)

    def get_embedding(
        self,
        text: str,
    ) -> List[float]:
        text = text.replace("\n", " ")
        return self.embedding.embed_query(text)

    def get_docs_embedding(
        self,
        docs: List[str],
    ) -> List[List[float]]:
        return self.embedding.embed_documents(docs)

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        return round(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), 4)

    def similar_tables_based_on_few_shot_examples(self, df: pd.DataFrame) -> List[str]:
        most_similar_tables = set()
        if self.few_shot_examples is not None:
            for example in self.few_shot_examples:
                try:
                    tables = Parser(example["sql"]).tables
                except Exception as e:
                    logger.error(f"Error parsing SQL: {str(e)}")
                for table in tables:
                    found_tables = df[df.table_name == table]
                    for _, row in found_tables.iterrows():
                        most_similar_tables.add((row["schema_name"], row["table_name"]))
            df.drop(
                df[
                    df.table_name.isin([table[1] for table in most_similar_tables])
                ].index,
                inplace=True,
            )
        return most_similar_tables

    @catch_exceptions()
    def _run(
        self,
        user_question: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        question_embedding = self.get_embedding(user_question)
        table_representations = []
        for table in self.db_scan:
            col_rep = ""
            for column in table.columns:
                if column.description:
                    col_rep += f"{column.name}: {column.description}, "
                else:
                    col_rep += f"{column.name}, "
            if table.description:
                table_rep = f"Table {table.table_name} contain columns: [{col_rep}], this tables has: {table.description}"
            else:
                table_rep = f"Table {table.table_name} contain columns: [{col_rep}]"
            table_representations.append(
                [table.schema_name, table.table_name, table_rep]
            )
        df = pd.DataFrame(
            table_representations,
            columns=["schema_name", "table_name", "table_representation"],
        )
        df["table_embedding"] = self.get_docs_embedding(df.table_representation)
        df["similarities"] = df.table_embedding.apply(
            lambda x: self.cosine_similarity(x, question_embedding)
        )
        df = df.sort_values(by="similarities", ascending=True)
        df = df.tail(TOP_TABLES)
        most_similar_tables = self.similar_tables_based_on_few_shot_examples(df)
        table_relevance = ""
        for _, row in df.iterrows():
            if row["schema_name"] is not None:
                table_name = row["schema_name"] + "." + row["table_name"]
            else:
                table_name = row["table_name"]
            table_relevance += (
                f'Table: `{table_name}`, relevance score: {row["similarities"]}\n'
            )
        if len(most_similar_tables) > 0:
            for table in most_similar_tables:
                if table[0] is not None:
                    table_name = table[0] + "." + table[1]
                else:
                    table_name = table[1]
                table_relevance += f"Table: `{table_name}`, relevance score: {max(df['similarities'])}\n"
        return table_relevance

    async def _arun(
        self,
        user_question: str = "",
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        raise NotImplementedError("TablesSQLDatabaseTool does not support async")


class QuerySQLDataBaseTool(BaseSQLDatabaseTool, BaseTool):
    name = "SqlDbQuery"
    description = """
    Input: A SQL query between ```sql and ``` tags.
    Output: Result from the database or an error message if the query is incorrect.
    Use this tool to execute the SQL query on the database, and return the results.
    Add newline after both ```sql and ``` tags.
    """
    args_schema: Type[BaseModel] = SQLInput

    @catch_exceptions()
    def _run(
        self,
        query: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Execute the query, return the results or an error message."""
        query = replace_unprocessable_characters(query)
        if "```sql" in query:
            query = query.replace("```sql", "").replace("```", "")

        try:
            return run_with_timeout(
                self.db.run_sql,
                args=(query,),
                kwargs={"top_k": TOP_K},
                timeout_duration=int(os.getenv("SQL_EXECUTION_TIMEOUT", "60")),
            )[0]
        except TimeoutError:
            return "SQL query execution time exceeded, proceed without query execution"

    async def _arun(
        self,
        query: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        raise NotImplementedError("QuerySQLDataBaseTool does not support async")


class GenerateSQL(BaseSQLDatabaseTool, BaseTool):
    name = "GenerateSQL"
    description = """
    Input: user question.
    Output: SQL query.
    Use this tool to a generate SQL queries.
    Pass the user question as input to the tool.
    """
    finetuning_model_id: str = Field(exclude=True)
    model_name: str = Field(exclude=True)
    args_schema: Type[BaseModel] = QuestionInput
    db_scan: List[TableDescription]
    api_key: str = Field(exclude=True)
    openai_fine_tuning: OpenAIFineTuning = Field(exclude=True)
    embedding: OpenAIEmbeddings = Field(exclude=True)

    @catch_exceptions()
    def _run(
        self, question: str = "", run_manager: CallbackManagerForToolRun | None = None
    ) -> str:
        table_representations = []
        for table in self.db_scan:
            table_representations.append(
                self.openai_fine_tuning.create_table_representation(table)
            )
        table_embeddings = self.embedding.embed_documents(table_representations)
        system_prompt = (
            FINETUNING_SYSTEM_INFORMATION
            + self.openai_fine_tuning.format_dataset(
                self.db_scan,
                table_embeddings,
                question,
                OPENAI_FINETUNING_MODELS_WINDOW_SIZES[self.model_name] - 500,
            )
        )
        user_prompt = "User Question: " + question + "\n SQL: "
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.finetuning_model_id,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        returned_sql = response.choices[0].message.content
        return f"```sql\n{returned_sql}```"

    async def _arun(
        self,
        tool_input: str = "",
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        raise NotImplementedError("GenerateSQL tool does not support async")


class SchemaSQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):
    name = "DbSchema"
    description = """
    Input: Comma-separated list of tables.
    Output: Schema of the specified tables.
    Use this tool to find the schema of the specified tables, if you are unsure about the schema of the tables when editing the SQL query.
    Example Input: table1, table2, table3
    """
    db_scan: List[TableDescription]

    @catch_exceptions()
    def _run(
        self,
        table_names: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        table_names_list = table_names.split(", ")
        processed_table_names = []
        for table in table_names_list:
            formatted_table = replace_unprocessable_characters(table)
            if "." in formatted_table:
                processed_table_names.append(formatted_table.split(".")[1])
            else:
                processed_table_names.append(formatted_table)
        tables_schema = ""
        for table in self.db_scan:
            if table.table_name in processed_table_names:
                tables_schema += "```sql\n"
                tables_schema += table.table_schema + "\n"
                descriptions = []
                if table.description is not None:
                    if table.schema_name:
                        table_name = f"{table.schema_name}.{table.table_name}"
                    else:
                        table_name = table.table_name
                    descriptions.append(f"Table `{table_name}`: {table.description}\n")
                    for column in table.columns:
                        if column.description is not None:
                            descriptions.append(
                                f"Column `{column.name}`: {column.description}\n"
                            )
                if len(descriptions) > 0:
                    tables_schema += f"/*\n{''.join(descriptions)}*/\n"
                tables_schema += "```\n"
        if tables_schema == "":
            tables_schema += "Tables not found in the database"
        return tables_schema

    async def _arun(
        self,
        table_name: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        raise NotImplementedError("SchemaSQLDatabaseTool does not support async")
