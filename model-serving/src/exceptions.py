"""
Custom exception classes and error handling for CarlaRL Policy-as-a-Service.

This module defines application-specific exceptions and provides utilities
for consistent error handling across the serving infrastructure.
"""

import logging
import time
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError

logger = logging.getLogger(__name__)


class CarlaRLServingException(Exception):
    """Base exception class for all CarlaRL serving errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "CARLA_RL_ERROR",
        details: dict[str, Any] | None = None,
        status_code: int = 500
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.status_code = status_code
        self.timestamp = time.time()
        super().__init__(self.message)


class ModelLoadingError(CarlaRLServingException):
    """Raised when model artifacts cannot be loaded or are corrupted."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            message=message,
            error_code="MODEL_LOADING_ERROR",
            details=details,
            status_code=500
        )


class ArtifactValidationError(CarlaRLServingException):
    """Raised when model artifacts fail integrity validation."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            message=message,
            error_code="ARTIFACT_VALIDATION_ERROR",
            details=details,
            status_code=422
        )


class InferenceError(CarlaRLServingException):
    """Raised when inference fails due to model or input issues."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            message=message,
            error_code="INFERENCE_ERROR",
            details=details,
            status_code=500
        )


class PreprocessingError(CarlaRLServingException):
    """Raised when preprocessing pipeline fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            message=message,
            error_code="PREPROCESSING_ERROR",
            details=details,
            status_code=422
        )


class ValidationError(CarlaRLServingException):
    """Raised when input validation fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=details,
            status_code=422
        )


class ServiceUnavailableError(CarlaRLServingException):
    """Raised when service is temporarily unavailable."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            message=message,
            error_code="SERVICE_UNAVAILABLE",
            details=details,
            status_code=503
        )


class ResourceExhaustedError(CarlaRLServingException):
    """Raised when system resources are exhausted."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            message=message,
            error_code="RESOURCE_EXHAUSTED",
            details=details,
            status_code=429
        )


def create_error_response(
    error: Exception,
    request_id: str | None = None
) -> dict[str, Any]:
    """
    Create standardized error response dictionary.

    Args:
        error: Exception instance
        request_id: Optional request ID for tracing

    Returns:
        Dictionary containing error information
    """
    if isinstance(error, CarlaRLServingException):
        return {
            "error": error.error_code,
            "message": error.message,
            "details": {
                **error.details,
                **({"request_id": request_id} if request_id else {})
            },
            "timestamp": error.timestamp
        }
    elif isinstance(error, PydanticValidationError):
        return {
            "error": "VALIDATION_ERROR",
            "message": "Input validation failed",
            "details": {
                "validation_errors": [
                    {
                        "field": ".".join(str(loc) for loc in err["loc"]),
                        "message": err["msg"],
                        "type": err["type"]
                    }
                    for err in error.errors()
                ],
                **({"request_id": request_id} if request_id else {})
            },
            "timestamp": time.time()
        }
    elif isinstance(error, HTTPException):
        return {
            "error": "HTTP_ERROR",
            "message": error.detail,
            "details": {
                "status_code": error.status_code,
                **({"request_id": request_id} if request_id else {})
            },
            "timestamp": time.time()
        }
    else:
        # Generic exception handling
        return {
            "error": "INTERNAL_ERROR",
            "message": "An unexpected error occurred",
            "details": {
                "error_type": type(error).__name__,
                **({"request_id": request_id} if request_id else {})
            },
            "timestamp": time.time()
        }


async def carla_rl_exception_handler(
    request: Request,
    exc: CarlaRLServingException
) -> JSONResponse:
    """
    FastAPI exception handler for CarlaRLServingException.

    Args:
        request: FastAPI request object
        exc: CarlaRLServingException instance

    Returns:
        JSONResponse with error details
    """
    request_id = getattr(request.state, "request_id", None)

    # Log the error for debugging
    logger.error(
        f"CarlaRL serving error: {exc.error_code} - {exc.message}",
        extra={
            "error_code": exc.error_code,
            "details": exc.details,
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method
        }
    )

    error_response = create_error_response(exc, request_id)
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response
    )


async def validation_exception_handler(
    request: Request,
    exc: PydanticValidationError
) -> JSONResponse:
    """
    FastAPI exception handler for Pydantic validation errors.

    Args:
        request: FastAPI request object
        exc: PydanticValidationError instance

    Returns:
        JSONResponse with validation error details
    """
    request_id = getattr(request.state, "request_id", None)

    # Log validation errors
    logger.warning(
        f"Validation error: {len(exc.errors())} validation failures",
        extra={
            "validation_errors": exc.errors(),
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method
        }
    )

    error_response = create_error_response(exc, request_id)
    return JSONResponse(
        status_code=422,
        content=error_response
    )


async def http_exception_handler(
    request: Request,
    exc: HTTPException
) -> JSONResponse:
    """
    FastAPI exception handler for HTTP exceptions.

    Args:
        request: FastAPI request object
        exc: HTTPException instance

    Returns:
        JSONResponse with HTTP error details
    """
    request_id = getattr(request.state, "request_id", None)

    # Log HTTP errors (but not 4xx client errors at ERROR level)
    log_level = logging.ERROR if exc.status_code >= 500 else logging.WARNING
    logger.log(
        log_level,
        f"HTTP error {exc.status_code}: {exc.detail}",
        extra={
            "status_code": exc.status_code,
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method
        }
    )

    error_response = create_error_response(exc, request_id)
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response
    )


async def generic_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """
    FastAPI exception handler for unexpected exceptions.

    Args:
        request: FastAPI request object
        exc: Exception instance

    Returns:
        JSONResponse with generic error details
    """
    request_id = getattr(request.state, "request_id", None)

    # Log unexpected errors at ERROR level
    logger.error(
        f"Unexpected error: {type(exc).__name__} - {str(exc)}",
        extra={
            "error_type": type(exc).__name__,
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method
        },
        exc_info=True  # Include stack trace
    )

    error_response = create_error_response(exc, request_id)
    return JSONResponse(
        status_code=500,
        content=error_response
    )


# Exception handler mapping for FastAPI
EXCEPTION_HANDLERS = {
    CarlaRLServingException: carla_rl_exception_handler,
    PydanticValidationError: validation_exception_handler,
    HTTPException: http_exception_handler,
    Exception: generic_exception_handler,
}
