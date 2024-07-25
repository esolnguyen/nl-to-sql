import datetime
import difflib
import logging
import os
from typing import List
import numpy as np
import pandas as pd
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.base import BaseTool
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
from sql_metadata import Parser
from sqlalchemy.exc import SQLAlchemyError
from llm_agents import SQLGenerator, replace_unprocessable_characters
from models.db_description import TableDescription
from utils.custom_error import catch_exceptions
from utils.sql_database import SQLDatabase
from utils.timeout import run_with_timeout

logger = logging.getLogger(__name__)


TOP_K = SQLGenerator.get_upper_bound_limit()
EMBEDDING_MODEL = "text-embedding-3-large"
TOP_TABLES = 20


class BaseSQLDatabaseTool(BaseModel):
    db: SQLDatabase = Field(exclude=True)
    context: List[dict] | None = Field(exclude=True, default=None)

    class Config(BaseTool.Config):
        arbitrary_types_allowed = True
        extra = "allow"


class SystemTime(BaseSQLDatabaseTool, BaseTool):
    name = "SystemTime"
    description = """
    Input is an empty string, output is the current data and time.
    Always use this tool before generating a query if there is any time or date in the given question.
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
        raise NotImplementedError("GetCurrentTimeTool does not support async")


class QuerySQLDataBaseTool(BaseSQLDatabaseTool, BaseTool):
    name = "SqlDbQuery"
    description = """
    Input: A well-formed multi-line SQL query between ```sql and ``` tags.
    Output: Result from the database or an error message if the query is incorrect.
    If an error occurs, rewrite the query and retry.
    Use this tool to execute SQL queries.
    Add newline after both ```sql and ``` tags.
    """

    @catch_exceptions()
    def _run(
        self,
        query: str,
        top_k: int = TOP_K,
        run_manager: CallbackManagerForToolRun | None = None
    ) -> str:
        query = replace_unprocessable_characters(query)
        if "```sql" in query:
            query = query.replace("```sql", "").replace("```", "")

        try:
            return run_with_timeout(
                self.db.run_sql,
                args=(query,),
                kwargs={"top_k": top_k},
                timeout_duration=int(os.getenv("SQL_EXECUTION_TIMEOUT", "60")),
            )[0]
        except TimeoutError:
            return "SQL query execution time exceeded, proceed without query execution"

    async def _arun(
        self,
        query: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        raise NotImplementedError(
            "QuerySQLDataBaseTool does not support async")


class GetUserInstructions(BaseSQLDatabaseTool, BaseTool):
    name = "GetAdminInstructions"
    description = """
    Input: is an empty string.
    Output: Database admin instructions before generating the SQL query.
    The generated SQL query MUST follow the admin instructions even it contradicts with the given question.
    """
    instructions: List[dict]

    @catch_exceptions()
    def _run(
        self,
        tool_input: str = "",
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        response = "Admin: All of the generated SQL queries must follow the below instructions:\n"
        for index, instruction in enumerate(self.instructions):
            response += f"{index + 1}) {instruction['instruction']}\n"
        return response

    async def _arun(
        self,
        tool_input: str = "",
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        raise NotImplementedError("GetUserInstructions does not support async")


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
                        most_similar_tables.add(
                            (row["schema_name"], row["table_name"]))
            df.drop(
                df[
                    df.table_name.isin([table[1]
                                       for table in most_similar_tables])
                ].index,
                inplace=True,
            )
        return most_similar_tables

    @catch_exceptions()
    def _run(
        self,
        user_question: str,
        run_manager: CallbackManagerForToolRun | None = None
    ) -> str:
        question_embedding = self.get_embedding(user_question)
        table_representations = []
        for table in self.db_scan:
            col_rep = ""
            for column in table.columns:
                if column.description is not None:
                    col_rep += f"{column.name}: {column.description}, "
                else:
                    col_rep += f"{column.name}, "
            if table.description is not None:
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
        df["table_embedding"] = self.get_docs_embedding(
            df.table_representation)
        df["similarities"] = df.table_embedding.apply(
            lambda x: self.cosine_similarity(x, question_embedding)
        )
        df = df.sort_values(by="similarities", ascending=True)
        df = df.tail(TOP_TABLES)
        most_similar_tables = self.similar_tables_based_on_few_shot_examples(
            df)
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
        raise NotImplementedError(
            "TablesSQLDatabaseTool does not support async")


class ColumnEntityChecker(BaseSQLDatabaseTool, BaseTool):
    name = "DbColumnEntityChecker"
    description = """
    Input: Column name and its corresponding table, and an entity.
    Output: cell-values found in the column similar to the given entity.
    Use this tool to get cell values similar to the given entity in the given column.

    Example Input: table1 -> column2, entity
    """
    db_scan: List[TableDescription]
    is_multiple_schema: bool

    def find_similar_strings(
        self, input_list: List[tuple], target_string: str, threshold=0.4
    ):
        similar_strings = []
        for item in input_list:
            similarity = difflib.SequenceMatcher(
                None, str(item[0]).strip().lower(), target_string.lower()
            ).ratio()
            if similarity >= threshold:
                similar_strings.append((str(item[0]).strip(), similarity))
        similar_strings.sort(key=lambda x: x[1], reverse=True)
        return similar_strings[:25]

    @catch_exceptions()
    def _run(
        self,
        tool_input: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            schema, entity = tool_input.split(",")
            table_name, column_name = schema.split("->")
            table_name = replace_unprocessable_characters(table_name)
            column_name = replace_unprocessable_characters(column_name).strip()
            if "." not in table_name and self.is_multiple_schema:
                raise Exception(
                    "Table name should be in the format schema_name.table_name"
                )
        except ValueError:
            return "Invalid input format, use following format: table_name -> column_name, entity (entity should be a string without ',')"
        search_pattern = f"%{entity.strip().lower()}%"
        search_query = f"SELECT DISTINCT {column_name} FROM {table_name} WHERE {column_name} ILIKE :search_pattern"
        try:
            search_results = self.db.engine.execute(
                search_query, {"search_pattern": search_pattern}
            ).fetchall()
            search_results = search_results[:25]
        except SQLAlchemyError:
            search_results = []
        distinct_query = (
            f"SELECT DISTINCT {column_name} FROM {table_name}"
        )
        results = self.db.engine.execute(distinct_query).fetchall()
        results = self.find_similar_strings(results, entity)
        similar_items = "Similar items:\n"
        already_added = {}
        for item in results:
            similar_items += f"{item[0]}\n"
            already_added[item[0]] = True
        if len(search_results) > 0:
            for item in search_results:
                if item[0] not in already_added:
                    similar_items += f"{item[0]}\n"
        return similar_items

    async def _arun(
        self,
        tool_input: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        raise NotImplementedError("ColumnEntityChecker does not support async")


class SchemaSQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):
    name = "DbRelevantTablesSchema"
    description = """
    Input: Comma-separated list of tables.
    Output: Schema of the specified tables.
    Use this tool to discover all columns of the relevant tables and identify potentially relevant columns.

    Example Input: table1, table2, table3
    """
    db_scan: List[TableDescription]

    @catch_exceptions()
    def _run(
        self,
        table_names: str,
        run_manager: CallbackManagerForToolRun | None = None
    ) -> str:
        table_names_list = table_names.split(", ")
        processed_table_names = []
        for table in table_names_list:
            formatted_table = replace_unprocessable_characters(table)
            if "." in formatted_table:
                processed_table_names.append(formatted_table.split(".")[1])
            else:
                processed_table_names.append(formatted_table)
        tables_schema = "```sql\n"
        for table in self.db_scan:
            if table.table_name in processed_table_names:
                tables_schema += table.table_schema + "\n"
                descriptions = []
                if table.description is not None:
                    if table.schema_name:
                        table_name = f"{table.schema_name}.{table.table_name}"
                    else:
                        table_name = table.table_name
                    descriptions.append(
                        f"Table `{table_name}`: {table.description}\n")
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
        raise NotImplementedError(
            "SchemaSQLDatabaseTool does not support async")


class InfoRelevantColumns(BaseSQLDatabaseTool, BaseTool):
    name = "DbRelevantColumnsInfo"
    description = """
    Input: Comma-separated list of potentially relevant columns with their corresponding table.
    Output: Information about the values inside the columns and their descriptions.
    Use this tool to gather details about potentially relevant columns. then, filter them, and identify the relevant ones.

    Example Input: table1 -> column1, table1 -> column2, table2 -> column1
    """
    db_scan: List[TableDescription]

    @catch_exceptions()
    def _run(
        self,
        column_names: str,
        run_manager: CallbackManagerForToolRun | None = None
    ) -> str:
        """Get the column level information."""
        items_list = column_names.split(", ")
        column_full_info = ""
        for item in items_list:
            if " -> " in item:
                table_name, column_name = item.split(" -> ")
                if "." in table_name:
                    table_name = table_name.split(".")[1]
                table_name = replace_unprocessable_characters(table_name)
                column_name = replace_unprocessable_characters(column_name)
                found = False
                for table in self.db_scan:
                    if table_name == table.table_name:
                        col_info = ""
                        for column in table.columns:
                            if column_name == column.name:
                                found = True
                                col_info += f"Description: {column.description},"
                                if column.low_cardinality:
                                    col_info += f" categories = {column.categories},"
                        col_info += " Sample rows: "
                        if found:
                            for row in table.examples:
                                col_info += row[column_name] + ", "
                            col_info = col_info[:-2]
                            if table.schema_name:
                                schema_table = f"{table.schema_name}.{table.table_name}"
                            else:
                                schema_table = table.table_name
                            column_full_info += f"Table: {schema_table}, column: {column_name}, additional info: {col_info}\n"
            else:
                return "Malformed input, input should be in the following format Example Input: table1 -> column1, table1 -> column2, table2 -> column1"
            if not found:
                column_full_info += f"Table: {table_name}, column: {column_name} not found in database\n"
        return column_full_info

    async def _arun(
        self,
        table_name: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        raise NotImplementedError(
            "InfoRelevantColumnsTool does not support async")


class GetFewShotExamples(BaseSQLDatabaseTool, BaseTool):
    name = "FewshotExamplesRetriever"
    description = """
    Input: Number of required Question/SQL pairs.
    Output: List of similar Question/SQL pairs related to the given question.
    Use this tool to fetch previously asked Question/SQL pairs as examples for improving SQL query generation.
    For complex questions, request more examples to gain a better understanding of tables and columns and the SQL keywords to use.
    If the given question is very similar to one of the retrieved examples, it is recommended to use the same SQL query and modify it slightly to fit the given question.
    Always use this tool first and before any other tool!
    """
    few_shot_examples: List[dict]

    @catch_exceptions()
    def _run(
        self,
        number_of_samples: str,
        run_manager: CallbackManagerForToolRun | None = None
    ) -> str:
        if number_of_samples.strip().isdigit():
            number_of_samples = int(number_of_samples.strip())
        else:
            return "Action input for the fewshot_examples_retriever tool should be an integer"
        returned_output = ""
        for example in self.few_shot_examples[:number_of_samples]:
            returned_output += f"Question: {example['prompt_text']} \n"
            returned_output += f"```sql\n{example['sql']}\n```\n"
        if returned_output == "":
            returned_output = "No previously asked Question/SQL pairs are available"
        return returned_output

    async def _arun(
        self,
        number_of_samples: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        raise NotImplementedError(
            "GetFewShotExamplesTool does not support async")
