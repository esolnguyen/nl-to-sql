from functools import wraps
import logging
from typing import Any, Callable
from fastapi.responses import JSONResponse
from src.app.constants.error_mapping import ERROR_MAPPING
import openai
from google.api_core.exceptions import GoogleAPIError
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


class CustomError(Exception):
    def __init__(self, message, description=None):
        super().__init__(message)
        self.description = description


class SchemaNotSupportedError(CustomError):
    pass


class PromptNotFoundError(CustomError):
    pass


class DatabaseConnectionNotFoundError(CustomError):
    pass


class SQLGenerationError(CustomError):
    pass


class EmptySQLGenerationError(CustomError):
    pass


class FinetuningNotAvailableError(CustomError):
    pass


def error_response(error, detail: dict, default_error_code=""):
    error_code = ERROR_MAPPING.get(error.__class__.__name__, default_error_code)
    description = getattr(error, "description", None)
    logger.error(
        f"Error code: {error_code}, message: {error}, description: {description}, detail: {detail}"
    )

    detail.pop("metadata", None)

    return JSONResponse(
        status_code=400,
        content={
            "error_code": error_code,
            "message": str(error),
            "description": description,
            "detail": detail,
        },
    )


def stream_error_response(error, detail: dict, default_error_code=""):
    error_code = ERROR_MAPPING.get(error.__class__.__name__, default_error_code)
    description = getattr(error, "description", None)
    logger.error(
        f"Error code: {error_code}, message: {error}, description: {description}, detail: {detail}"
    )

    detail.pop("metadata", None)

    return {
        "error_code": error_code,
        "message": str(error),
        "description": description,
        "detail": detail,
    }


def catch_exceptions():
    def decorator(fn: Callable[[str], str]) -> Callable[[str], str]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except openai.AuthenticationError as e:
                return f"OpenAI API authentication error: {e}"
            except openai.RateLimitError as e:
                return f"OpenAI API request exceeded rate limit: {e}"
            except openai.BadRequestError as e:
                return f"OpenAI API request timed out: {e}"
            except openai.APIResponseValidationError as e:
                return f"OpenAI API response is invalid: {e}"
            except openai.OpenAIError as e:
                return f"OpenAI API returned an error: {e}"
            except GoogleAPIError as e:
                return f"Google API returned an error: {e}"
            except SQLAlchemyError as e:
                return f"Error: {e}"

        return wrapper

    return decorator
