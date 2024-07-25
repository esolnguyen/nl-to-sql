from typing import List
from llm_agents.toolkit import AgentToolkit
from langchain.tools.base import BaseTool
from llm_agents.toolkit.toolkits import ColumnEntityChecker, GetFewShotExamples, GetUserInstructions, InfoRelevantColumns, QuerySQLDataBaseTool, SchemaSQLDatabaseTool, SystemTime, TablesSQLDatabaseTool
from overrides import override


class SQLGenerationAgentToolkit(AgentToolkit):
 
    @override
    def get_tools(self) -> List[BaseTool]:
        tools = []
        query_sql_db_tool = QuerySQLDataBaseTool(
            db=self.db, context=self.context)
        tools.append(query_sql_db_tool)
        if self.instructions is not None:
            tools.append(
                GetUserInstructions(
                    db=self.db, context=self.context, instructions=self.instructions
                )
            )
        get_current_datetime = SystemTime(db=self.db, context=self.context)
        tools.append(get_current_datetime)
        tables_sql_db_tool = TablesSQLDatabaseTool(
            db=self.db,
            context=self.context,
            db_scan=self.db_scan,
            embedding=self.embedding,
            few_shot_examples=self.few_shot_examples,
        )
        tools.append(tables_sql_db_tool)
        schema_sql_db_tool = SchemaSQLDatabaseTool(
            db=self.db, context=self.context, db_scan=self.db_scan
        )
        tools.append(schema_sql_db_tool)
        info_relevant_tool = InfoRelevantColumns(
            db=self.db, context=self.context, db_scan=self.db_scan
        )
        tools.append(info_relevant_tool)
        column_sample_tool = ColumnEntityChecker(
            db=self.db,
            context=self.context,
            db_scan=self.db_scan,
            is_multiple_schema=self.is_multiple_schema,
        )
        tools.append(column_sample_tool)
        if self.few_shot_examples is not None:
            get_fewshot_examples_tool = GetFewShotExamples(
                db=self.db,
                context=self.context,
                few_shot_examples=self.few_shot_examples,
            )
            tools.append(get_fewshot_examples_tool)
        return tools
