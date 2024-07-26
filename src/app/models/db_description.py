from datetime import datetime, timezone
from enum import Enum
from typing import Any
from bson import ObjectId
from pydantic import BaseModel, Field, validator
from app.models.db_conntection import DBConnectionValidation
from bson.errors import InvalidId


class RefreshTableDescriptionRequest(DBConnectionValidation):
    pass


class ScannerRequest(BaseModel):
    ids: list[str] | None
    metadata: dict | None

    @validator("ids")
    def ids_validation(cls, ids: list = None):
        try:
            for id in ids:
                ObjectId(id)
        except InvalidId:
            raise ValueError("Must be a valid ObjectId")
        return ids


class ForeignKeyDetail(BaseModel):
    field_name: str
    reference_table: str


class ColumnDetail(BaseModel):
    name: str
    is_primary_key: bool = False
    data_type: str = "str"
    description: str | None
    low_cardinality: bool = False
    categories: list[Any] | None
    foreign_key: ForeignKeyDetail | None


class TableDescriptionStatus(Enum):
    NOT_SCANNED = "NOT_SCANNED"
    SYNCHRONIZING = "SYNCHRONIZING"
    DEPRECATED = "DEPRECATED"
    SCANNED = "SCANNED"
    FAILED = "FAILED"


class ColumnDescriptionRequest(BaseModel):
    name: str
    description: str | None
    is_primary_key: bool | None
    data_type: str | None
    low_cardinality: bool | None
    categories: list[str] | None
    foreign_key: ForeignKeyDetail | None


class TableDescriptionRequest(BaseModel):
    description: str | None
    columns: list[ColumnDescriptionRequest] | None
    metadata: dict | None


class TableDescription(BaseModel):
    id: str | None
    db_connection_id: str
    schema_name: str | None
    table_name: str
    description: str | None
    table_schema: str | None
    columns: list[ColumnDetail] = []
    examples: list = []
    last_schema_sync: datetime | None
    status: str = TableDescriptionStatus.SCANNED.value
    error_message: str | None
    metadata: dict | None
    created_at: datetime = Field(default_factory=datetime.now)

    @validator("last_schema_sync", pre=True)
    def parse_datetime_with_timezone(cls, value):
        if not value:
            return None
        return value.replace(tzinfo=timezone.utc)
