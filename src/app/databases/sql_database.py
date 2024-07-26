import logging
import os
import re
from typing import List
from urllib.parse import unquote
import sqlparse
from sqlalchemy import MetaData, create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sshtunnel import SSHTunnelForwarder
from app.models.sql_generation import SQLGeneration
from app.utils.timeout import run_with_timeout
from app.utils.aws_s3 import AWSS3
from app.utils.encrypt import FernetEncrypt
from app.utils.custom_error import CustomError
from app.models.db_conntection import DatabaseConnection

logger = logging.getLogger(__name__)


class SQLInjectionError(CustomError):
    pass


class InvalidDBConnectionError(CustomError):
    pass


class EmptyDBError(CustomError):
    pass


class SSHInvalidDatabaseConnectionError(CustomError):
    pass


class DbConnectionService:
    db_connections = {}

    @staticmethod
    def add(uri, engine):
        DbConnectionService.db_connections[uri] = engine


class SQLDatabase:
    def __init__(self, engine: Engine):
        self._engine = engine

    @property
    def engine(self) -> Engine:
        return self._engine

    @classmethod
    def from_uri(
        cls, database_uri: str, engine_args: dict | None = None
    ) -> "SQLDatabase":
        _engine_args = engine_args or {}
        if database_uri.lower().startswith("duckdb"):
            config = {"autoload_known_extensions": False}
            _engine_args["connect_args"] = {"config": config}
        engine = create_engine(database_uri, **_engine_args)
        return cls(engine)

    @classmethod
    def get_sql_engine(
        cls, database_info: DatabaseConnection, refresh_connection=False
    ) -> "SQLDatabase":
        logger.info(f"Connecting db: {database_info.id}")
        try:
            if (
                database_info.id
                and database_info.id in DbConnectionService.db_connections
                and not refresh_connection
            ):
                sql_database = DbConnectionService.db_connections[database_info.id]
                sql_database.engine.connect()
                return sql_database
        except OperationalError:
            pass

        fernet_encrypt = FernetEncrypt()
        try:
            if database_info.use_ssh:
                engine = cls.from_uri_ssh(database_info)
                DbConnectionService.add(database_info.id, engine)
                return engine
        except Exception as e:
            raise SSHInvalidDatabaseConnectionError(
                "Invalid SSH connection", description=str(e)
            ) from e
        try:
            db_uri = unquote(fernet_encrypt.decrypt(database_info.connection_uri))

            file_path = database_info.path_to_credentials_file
            if file_path and file_path.lower().startswith("s3"):
                s3 = AWSS3()
                file_path = s3.download(file_path)

            if db_uri.lower().startswith("bigquery"):
                db_uri = db_uri + f"?credentials_path={file_path}"

            engine = cls.from_uri(db_uri)
            engine.engine.connect()
            DbConnectionService.add(database_info.id, engine)
        except Exception as e:
            raise InvalidDBConnectionError(
                f"Unable to connect to db: {database_info.alias}", description=str(e)
            )
        return engine

    @classmethod
    def extract_parameters(cls, input_string):
        pattern = r"([^:/]+)://([^:]+):([^@]+)@([^:/]+)(?::(\d+))?/([^/]+)"

        match = re.match(pattern, input_string)

        if match:
            driver = match.group(1)
            user = match.group(2)
            password = match.group(3)
            host = match.group(4)
            port = match.group(5)
            db = match.group(6) if match.group(6) else None

            return {
                "driver": driver,
                "user": user,
                "password": password,
                "host": host,
                "port": port if port else None,
                "db": db,
            }

        return None

    @classmethod
    def from_uri_ssh(cls, database_info: DatabaseConnection):
        file_path = database_info.path_to_credentials_file
        if file_path.lower().startswith("s3"):
            s3 = AWSS3()
            file_path = s3.download(file_path)

        fernet_encrypt = FernetEncrypt()
        db_uri = unquote(fernet_encrypt.decrypt(database_info.connection_uri))
        db_uri_obj = cls.extract_parameters(db_uri)
        ssh = database_info.ssh_settings

        server = SSHTunnelForwarder(
            (ssh.host, 22 if not ssh.port else int(ssh.port)),
            ssh_username=ssh.username,
            ssh_password=fernet_encrypt.decrypt(ssh.password),
            ssh_pkey=file_path,
            ssh_private_key_password=fernet_encrypt.decrypt(ssh.private_key_password),
            remote_bind_address=(
                db_uri_obj["host"],
                5432 if not db_uri_obj["port"] else int(db_uri_obj["port"]),
            ),
        )
        server.stop(force=True)
        server.start()
        local_port = str(server.local_bind_port)
        local_host = str(server.local_bind_host)

        return cls.from_uri(
            f"{db_uri_obj['driver']}://{db_uri_obj['user']}:{db_uri_obj['password']}@{local_host}:{local_port}/{db_uri_obj['db']}"
        )

    @classmethod
    def parser_to_filter_commands(cls, command: str) -> str:
        sensitive_keywords = [
            "DROP",
            "DELETE",
            "UPDATE",
            "INSERT",
            "GRANT",
            "REVOKE",
            "ALTER",
            "TRUNCATE",
            "MERGE",
            "EXECUTE",
            "CREATE",
        ]
        parsed_command = sqlparse.parse(command)

        for stmt in parsed_command:
            for token in stmt.tokens:
                if (
                    isinstance(token, sqlparse.sql.Token)
                    and token.normalized in sensitive_keywords
                ):
                    raise SQLInjectionError(
                        f"Sensitive SQL keyword '{token.normalized}' detected in the query."
                    )

        return command

    def run_sql(self, command: str, top_k: int = None) -> tuple[str, dict]:
        with self._engine.connect() as connection:
            command = self.parser_to_filter_commands(command)
            cursor = connection.execute(text(command))
            if cursor.returns_rows and top_k:
                result = cursor.fetchmany(top_k)
                return str(result), {"result": result}
            if cursor.returns_rows:
                result = cursor.fetchall()
                return str(result), {"result": result}
        return "", {}

    def get_tables_and_views(self) -> List[str]:
        inspector = inspect(self._engine)
        meta = MetaData(bind=self._engine)
        MetaData.reflect(meta, views=True)
        rows = inspector.get_table_names() + inspector.get_view_names()
        if len(rows) == 0:
            raise EmptyDBError("The db is empty it could be a permission issue")
        return [row.lower() for row in rows]

    @property
    def dialect(self) -> str:
        return self._engine.dialect.name


def format_error_message(
    sql_generation: SQLGeneration, error_message: str
) -> SQLGeneration:
    if error_message.find("[") > 0 and error_message.find("]") > 0:
        error_message = (
            error_message[0 : error_message.find("[")]
            + error_message[error_message.rfind("]") + 1 :]
        )
    sql_generation.status = "INVALID"
    sql_generation.error = error_message
    return sql_generation


def create_sql_query_status(
    db: SQLDatabase,
    query: str,
    sql_generation: SQLGeneration,
) -> SQLGeneration:
    if query == "":
        sql_generation.status = "INVALID"
        sql_generation.error = "Sorry, we couldn't generate an SQL from your prompt"
    else:
        try:
            query = db.parser_to_filter_commands(query)
            run_with_timeout(
                db.run_sql,
                args=(query,),
                timeout_duration=int(os.getenv("SQL_EXECUTION_TIMEOUT", "60")),
            )
            sql_generation.status = "VALID"
            sql_generation.error = None
        except TimeoutError:
            sql_generation = format_error_message(
                sql_generation,
                "The query execution exceeded the timeout.",
            )
        except SQLInjectionError as e:
            raise SQLInjectionError(
                "Sensitive SQL keyword detected in the query."
            ) from e
        except Exception as e:
            sql_generation = format_error_message(sql_generation, str(e))
    return sql_generation
