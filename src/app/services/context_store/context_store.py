import logging
from typing import List, Tuple

from overrides import override
from sql_metadata import Parser

from src.app.models.example_sql import ExampleSQL, ExampleSQLRequest
from src.app.models.prompt import Prompt
from repositories.db_connections import DatabaseConnectionRepository
from repositories.example_sqls import ExampleSQLRepository
from repositories.instructions import InstructionRepository
from services.context_store import ContextStore
from src.app.utils.custom_error import DatabaseConnectionNotFoundError
from src.app.utils.sql_utils import extract_the_schemas_from_sql
from config import System

logger = logging.getLogger(__name__)


class MalformedExampleSQLError(Exception):
    pass


class ContextStoreService(ContextStore):
    def __init__(self, system: System):
        super().__init__(system)

    @override
    def retrieve_context_for_question(
        self, prompt: Prompt, number_of_samples: int = 3
    ) -> Tuple[List[dict] | None, List[dict] | None]:
        logger.info(f"Getting context for {prompt.text}")
        closest_questions = self.vector_store.query(
            query_texts=[prompt.text],
            db_connection_id=prompt.db_connection_id,
            collection=self.example_sql_collection,
            num_results=number_of_samples,
        )

        samples = []
        example_sqls_repository = ExampleSQLRepository(self.db)
        for question in closest_questions:
            example_sql = example_sqls_repository.find_by_id(question["id"])
            if example_sql is not None:
                samples.append(
                    {
                        "prompt_text": example_sql.prompt_text,
                        "sql": example_sql.sql,
                        "score": question["score"],
                    }
                )
        if len(samples) == 0:
            samples = None
        instructions = []
        instruction_repository = InstructionRepository(self.db)
        all_instructions = instruction_repository.find_all()
        for instruction in all_instructions:
            if instruction.db_connection_id == prompt.db_connection_id:
                instructions.append(
                    {
                        "instruction": instruction.instruction,
                    }
                )
        if len(instructions) == 0:
            instructions = None

        return samples, instructions

    @override
    def add_example_sqls(
        self, example_sqls: List[ExampleSQLRequest]
    ) -> List[ExampleSQL]:
        example_sqls_repository = ExampleSQLRepository(self.db)
        db_connection_repository = DatabaseConnectionRepository(self.db)
        stored_example_sqls = []
        for record in example_sqls:
            try:
                Parser(record.sql).tables
            except Exception as e:
                raise MalformedExampleSQLError(
                    f"SQL {record.sql} is malformed. Please check the syntax."
                ) from e

            db_connection = db_connection_repository.find_by_id(record.db_connection_id)
            if not db_connection:
                raise DatabaseConnectionNotFoundError(
                    f"Database connection not found, {record.db_connection_id}"
                )

            if db_connection.schemas:
                schema_not_found = True
                used_schemas = extract_the_schemas_from_sql(record.sql)
                for schema in db_connection.schemas:
                    if schema in used_schemas:
                        schema_not_found = False
                        break
                if schema_not_found:
                    raise MalformedExampleSQLError(
                        f"SQL {record.sql} does not contain any of the schemas {db_connection.schemas}"
                    )

            prompt_text = record.prompt_text
            example_sql = ExampleSQL(
                prompt_text=prompt_text,
                sql=record.sql,
                db_connection_id=record.db_connection_id,
                metadata=record.metadata,
            )
            stored_example_sqls.append(example_sqls_repository.insert(example_sql))
        self.vector_store.add_records(stored_example_sqls, self.example_sql_collection)
        return stored_example_sqls

    @override
    def remove_example_sqls(self, ids: List) -> bool:
        """Removes the example sqls from the DB and the VectorDB"""
        example_sqls_repository = ExampleSQLRepository(self.db)
        for id in ids:
            self.vector_store.delete_record(
                collection=self.example_sql_collection, id=id
            )
            deleted = example_sqls_repository.delete_by_id(id)
            if deleted == 0:
                logger.warning(f"Example record with id {id} not found")
        return True
