from functools import lru_cache

from cryptography.fernet import Fernet

from config import Settings


@lru_cache(maxsize=1)
def _get_fernet() -> Fernet:
    return Fernet(Settings().require("encrypt_key"))


class FernetEncrypt:
    def __init__(self):
        self.fernet_key = _get_fernet()

    def encrypt(self, input: str) -> str:
        if not input:
            return ""
        return self.fernet_key.encrypt(input.encode()).decode("utf-8")

    def decrypt(self, input: str) -> str:
        if input == "":
            return ""
        return self.fernet_key.decrypt(input).decode("utf-8")
