from typing import List, Type
from pydantic import Field, BaseModel
from src.app.llm_agents.toolkit import AgentToolkit
from overrides import override
from langchain.tools.base import BaseTool
from src.app.llm_agents.toolkit.toolkits import BaseSQLDatabaseTool, QuerySQLDataBaseTool, SchemaSQLDatabaseTool, SystemTime, TablesSQLDatabaseTool
from src.app.models.db_description import TableDescription
from src.app.utils.custom_error import catch_exceptions


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
        self,
        question: str = "",
        run_manager: CallbackManagerForToolRun | None = None
    ) -> str:
        """Execute the query, return the results or an error message."""
        table_representations = []
        for table in self.db_scan:
            table_representations.append(
                self.openai_fine_tuning.create_table_representation(table)
            )
        table_embeddings = self.embedding.embed_documents(
            table_representations)
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


class FineTuningAgentToolkit(AgentToolkit):
    openai_fine_tuning: OpenAIFineTuning = Field(exclude=True)
    finetuning_model_id: str = Field(exclude=True)
    model_name: str = Field(exclude=True)
    api_key: str = Field(exclude=True)
    use_finetuned_model_only: bool = Field(exclude=True, default=None)

    @override
    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        tools = []
        if not self.use_finetuned_model_only:
            tools.append(SystemTime(db=self.db))
            tools.append(SchemaSQLDatabaseTool(
                db=self.db, db_scan=self.db_scan))
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
