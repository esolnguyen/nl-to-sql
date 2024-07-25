from bson.objectid import ObjectId
from models.example_sql import ExampleSQL

DB_COLLECTION = "example_sqls"


class ExampleSQLNotFoundError(Exception):
    pass


class ExampleSQLRepository:
    def __init__(self, storage):
        self.storage = storage

    def insert(self, example_sql: ExampleSQL) -> ExampleSQL:
        example_sql_dict = example_sql.dict(exclude={"id"})
        example_sql_dict["db_connection_id"] = str(
            example_sql.db_connection_id)
        example_sql.id = str(self.storage.insert_one(
            DB_COLLECTION, example_sql_dict))
        return example_sql

    def find_one(self, query: dict) -> ExampleSQL | None:
        row = self.storage.find_one(DB_COLLECTION, query)
        if not row:
            return None
        row["id"] = str(row["_id"])
        row["db_connection_id"] = str(row["db_connection_id"])
        return ExampleSQL(**row)

    def update(self, example_sql: ExampleSQL) -> ExampleSQL:
        example_sql_dict = example_sql.dict(exclude={"id"})
        example_sql_dict["db_connection_id"] = str(
            example_sql.db_connection_id)

        self.storage.update_or_create(
            DB_COLLECTION,
            {"_id": ObjectId(example_sql.id)},
            example_sql_dict,
        )
        return example_sql

    def find_by_id(self, id: str) -> ExampleSQL | None:
        row = self.storage.find_one(DB_COLLECTION, {"_id": ObjectId(id)})
        if not row:
            return None
        row["id"] = str(row["_id"])
        row["db_connection_id"] = str(row["db_connection_id"])
        return ExampleSQL(**row)

    def find_by(self, query: dict, page: int = 1, limit: int = 10) -> list[ExampleSQL]:
        rows = self.storage.find(DB_COLLECTION, query, page=page, limit=limit)
        example_sqls = []
        for row in rows:
            row["id"] = str(row["_id"])
            row["db_connection_id"] = str(row["db_connection_id"])
            example_sqls.append(ExampleSQL(**row))
        return example_sqls

    def find_all(self, page: int = 0, limit: int = 0) -> list[ExampleSQL]:
        rows = self.storage.find_all(DB_COLLECTION, page=page, limit=limit)
        example_sqls = []
        for row in rows:
            row["id"] = str(row["_id"])
            row["db_connection_id"] = str(row["db_connection_id"])
            example_sqls.append(ExampleSQL(**row))
        return example_sqls

    def delete_by_id(self, id: str) -> int:
        return self.storage.delete_by_id(DB_COLLECTION, id)
