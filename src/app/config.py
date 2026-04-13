import importlib
import inspect
from abc import ABC
from typing import Any, Dict, Type, TypeVar, cast

from dotenv import load_dotenv
from overrides import EnforceOverrides
from pydantic import BaseSettings, Field

load_dotenv()

_abstract_type_keys: Dict[str, str] = {
    "app.api.API": "api_impl",
    "app.services.db_scanner.Scanner": "db_scanner_impl",
    "app.databases.mongodb.NlToSQLDatabase": "db_impl",
    "app.services.context_store.ContextStore": "context_store_impl",
    "app.databases.vector.VectorDatabase": "vector_store_impl",
}


class Settings(BaseSettings):
    api_impl: str = Field("app.api.nl_to_sql_api.NLToSQLAPI", env="API_SERVER")
    db_scanner_impl: str = Field(
        "app.services.db_scanner.SqlAlchemyScannerService", env="DB_SCANNER"
    )
    db_impl: str = Field("app.databases.mongodb.mongo.MongoDB", env="NL_TO_SQL_DB")
    context_store_impl: str = Field(
        "app.services.context_store.context_store.ContextStoreService",
        env="CONTEXT_STORE",
    )
    vector_store_impl: str = Field(
        "app.databases.vector.chroma.Chroma", env="VECTOR_STORE"
    )

    db_name: str | None = Field(None, env="MONGODB_DB_NAME")
    db_uri: str | None = Field(None, env="MONGODB_URI")
    openai_api_key: str | None = Field(None, env="OPENAI_API_KEY")
    encrypt_key: str | None = Field(None, env="ENCRYPT_KEY")
    s3_custom_endpoint: str | None = Field(None, env="S3_CUSTOM_ENDPOINT")
    s3_bucket_name: str = Field("k2-core", env="S3_BUCKET_NAME")
    s3_region: str | None = Field("us-east-1", env="S3_REGION")
    s3_aws_access_key_id: str | None = Field(None, env="S3_AWS_ACCESS_KEY_ID")
    s3_aws_secret_access_key: str | None = Field(None, env="S3_AWS_SECRET_ACCESS_KEY")

    azure_api_key: str | None = Field(None, env="AZURE_API_KEY")
    embedding_model: str | None = Field(None, env="EMBEDDING_MODEL")
    azure_api_version: str | None = Field(None, env="AZURE_API_VERSION")
    only_store_csv_files_locally: bool = Field(
        False, env="ONLY_STORE_CSV_FILES_LOCALLY"
    )

    def require(self, key: str) -> Any:
        val = self[key]
        if val is None:
            raise ValueError(f"Missing required config value '{key}'")
        return val

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


T = TypeVar("T", bound="Component")


class Component(ABC, EnforceOverrides):
    _running: bool

    def __init__(self, system: "System"): 
        self._running = False

    def stop(self) -> None:
        """Idempotently stop this component's execution and free all associated
        resources."""
        self._running = False

    def start(self) -> None:
        """Idempotently start this component's execution"""
        self._running = True


class System(Component):
    settings: Settings
    _instances: Dict[Type[Component], Component]

    def __init__(self, settings: Settings):
        self.settings = settings
        self._instances = {}
        super().__init__(self)

    def instance(self, type: Type[T]) -> T:
        """Return an instance of the component type specified. If the system is running,
        the component will be started as well."""

        if inspect.isabstract(type):
            type_fqn = get_fqn(type)
            if type_fqn not in _abstract_type_keys:
                raise ValueError(f"Cannot instantiate abstract type: {type}")
            key = _abstract_type_keys[type_fqn]
            fqn = self.settings.require(key)
            type = get_class(fqn, type)

        if type not in self._instances:
            impl = type(self)
            self._instances[type] = impl
            if self._running:
                impl.start()

        inst = self._instances[type]
        return cast(T, inst)


C = TypeVar("C")


def get_class(fqn: str, type: Type[C]) -> Type[C]: 
    """Given a fully qualifed class name, import the module and return the class"""
    module_name, class_name = fqn.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cast(Type[C], cls)


def get_fqn(cls: Type[object]) -> str:
    """Given a class, return its fully qualified name"""
    return f"{cls.__module__}.{cls.__name__}"
