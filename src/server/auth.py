"""Lightweight API-key authentication.

A single shared key is read from the ``API_KEY`` environment variable
(via ``Settings``). When it is unset the dependency is a no-op so local
development keeps working, but a warning is logged so the gap is visible.
When it is set, every protected route requires a matching ``X-API-Key``
header (or ``Authorization: Bearer <key>``).
"""
import hmac
import logging

from fastapi import Header, HTTPException, status

from config import Settings

logger = logging.getLogger(__name__)

_API_KEY_WARNING_EMITTED = False


def _warn_once() -> None:
    global _API_KEY_WARNING_EMITTED  # noqa: PLW0603
    if not _API_KEY_WARNING_EMITTED:
        logger.warning(
            "API_KEY is not set — authentication is DISABLED. "
            "Do not expose this service publicly without setting API_KEY."
        )
        _API_KEY_WARNING_EMITTED = True


def make_api_key_dependency(settings: Settings):
    """Return a FastAPI dependency that enforces the configured API key."""

    expected = settings.api_key

    def verify_api_key(
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        authorization: str | None = Header(default=None),
    ) -> None:
        if not expected:
            _warn_once()
            return

        provided = x_api_key
        if not provided and authorization:
            scheme, _, token = authorization.partition(" ")
            if scheme.lower() == "bearer":
                provided = token

        if not provided or not hmac.compare_digest(provided, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
                headers={"WWW-Authenticate": "X-API-Key"},
            )

    return verify_api_key
