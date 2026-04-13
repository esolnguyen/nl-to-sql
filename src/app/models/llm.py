import os
from pydantic import BaseModel, BaseSettings, validator
from app.utils.encrypt import FernetEncrypt
from typing import Any


class LLMConfig(BaseModel):
    llm_name: str = os.getenv("LLM_NAME", "gpt-4-turbo-preview")
    api_base: str | None = None


class LLMCredentials(BaseSettings):
    organization_id: str | None
    api_key: str | None

    @validator("api_key", "organization_id", pre=True, always=True)
    def encrypt(cls, value: str):
        fernet_encrypt = FernetEncrypt()
        try:
            fernet_encrypt.decrypt(value)
            return value
        except Exception:
            return fernet_encrypt.encrypt(value)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)
