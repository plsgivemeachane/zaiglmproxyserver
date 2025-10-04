"""
Centralized Exception Handling System for GLM OpenAI-Compatible API Wrapper

This module provides a comprehensive exception hierarchy and error handling
framework for the GLM server application.
"""

import json
import traceback
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from flask import jsonify, request, Response
from pydantic import BaseModel


class ErrorCode(str, Enum):
    """Standardized error codes."""
    
    # Client errors (4xx)
    INVALID_REQUEST = "invalid_request"
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"
    NOT_FOUND = "not_found"
    METHOD_NOT_ALLOWED = "method_not_allowed"
    CONFLICT = "conflict"
    VALIDATION_ERROR = "validation_error"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    
    # Server errors (5xx)
    INTERNAL_SERVER_ERROR = "internal_server_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    GATEWAY_TIMEOUT = "gateway_timeout"
    INSUFFICIENT_STORAGE = "insufficient_storage"
    
    # GLM API specific errors
    GLM_API_ERROR = "glm_api_error"
    GLM_TIMEOUT = "glm_timeout"
    GLM_RATE_LIMIT = "glm_rate_limit"
    GLM_INVALID_MODEL = "glm_invalid_model"
    GLM_QUOTA_EXCEEDED = "glm_quota_exceeded"
    
    # Token management errors
    TOKEN_EXHAUSTION = "token_exhaustion"
    TOKEN_INVALID = "token_invalid"
    TOKEN_EXPIRED = "token_expired"
    TOKEN_ROTATION_FAILED = "token_rotation_failed"
    
    # Configuration errors
    CONFIG_ERROR = "config_error"
    CONFIG_VALIDATION_ERROR = "config_validation_error"
    
    # Processing errors
    RESPONSE_PROCESSING_ERROR = "response_processing_error"
    CONTENT_FILTERING_ERROR = "content_filtering_error"
    THINKING_TAG_PROCESSING_ERROR = "thinking_tag_processing_error"


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Error categories for better organization."""
    CLIENT_ERROR = "client_error"
    SERVER_ERROR = "server_error"
    EXTERNAL_API_ERROR = "external_api_error"
    CONFIGURATION_ERROR = "configuration_error"
    AUTHENTICATION_ERROR = "authentication_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    VALIDATION_ERROR = "validation_error"


class ErrorDetail(BaseModel):
    """Detailed error information."""
    
    type: str
    message: str
    code: ErrorCode
    severity: ErrorSeverity
    category: ErrorCategory
    details: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None
    documentation_url: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standardized error response format."""
    
    error: ErrorDetail
    request_id: str
    timestamp: datetime
    path: Optional[str] = None
    method: Optional[str] = None
    user_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "error": {
                "type": self.error.type,
                "message": self.error.message,
                "code": self.error.code.value,
                "severity": self.error.severity.value,
                "category": self.error.category.value,
                "details": self.error.details,
                "suggestions": self.error.suggestions,
                "documentation_url": self.error.documentation_url
            },
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "path": self.path,
            "method": self.method,
            "user_id": self.user_id
        }


class BaseGLMException(Exception):
    """Base exception class for all GLM server exceptions."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None,
        documentation_url: Optional[str] = None,
        http_status: int = 500
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.category = category
        self.details = details or {}
        self.suggestions = suggestions or []
        self.documentation_url = documentation_url
        self.http_status = http_status
        self.request_id = str(uuid4())
        self.timestamp = datetime.utcnow()
    
    def to_error_response(self) -> ErrorResponse:
        """Convert exception to ErrorResponse."""
        error_detail = ErrorDetail(
            type=self.__class__.__name__,
            message=self.message,
            code=self.error_code,
            severity=self.severity,
            category=self.category,
            details=self.details,
            suggestions=self.suggestions,
            documentation_url=self.documentation_url
        )
        
        return ErrorResponse(
            error=error_detail,
            request_id=self.request_id,
            timestamp=self.timestamp,
            path=getattr(request, 'path', None) if request else None,
            method=getattr(request, 'method', None) if request else None
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary."""
        return self.to_error_response().to_dict()


# Client Error Exceptions (4xx)

class ValidationError(BaseGLMException):
    """Request validation error."""
    
    def __init__(self, message: str, field_errors: Optional[Dict[str, List[str]]] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.VALIDATION_ERROR,
            details={"field_errors": field_errors} if field_errors else None,
            suggestions=["Check the request format and required fields"],
            http_status=400
        )


class AuthenticationError(BaseGLMException):
    """Authentication error."""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            message=message,
            error_code=ErrorCode.UNAUTHORIZED,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.AUTHENTICATION_ERROR,
            suggestions=["Provide a valid API key", "Check authentication headers"],
            http_status=401
        )


class AuthorizationError(BaseGLMException):
    """Authorization error."""
    
    def __init__(self, message: str = "Access forbidden"):
        super().__init__(
            message=message,
            error_code=ErrorCode.FORBIDDEN,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.AUTHENTICATION_ERROR,
            suggestions=["Check your permissions", "Contact administrator"],
            http_status=403
        )


class ResourceNotFoundError(BaseGLMException):
    """Resource not found error."""
    
    def __init__(self, resource: str, resource_id: str = None):
        message = f"{resource} not found"
        if resource_id:
            message += f": {resource_id}"
        
        super().__init__(
            message=message,
            error_code=ErrorCode.NOT_FOUND,
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.CLIENT_ERROR,
            details={"resource": resource, "resource_id": resource_id},
            suggestions=["Check the resource identifier", "Verify the resource exists"],
            http_status=404
        )


class RateLimitExceededError(BaseGLMException):
    """Rate limit exceeded error."""
    
    def __init__(self, limit: int, window: str, retry_after: Optional[int] = None):
        super().__init__(
            message=f"Rate limit exceeded: {limit} requests per {window}",
            error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.RATE_LIMIT_ERROR,
            details={"limit": limit, "window": window, "retry_after": retry_after},
            suggestions=["Reduce request frequency", "Implement backoff strategy"],
            http_status=429
        )


class PayloadTooLargeError(BaseGLMException):
    """Payload too large error."""
    
    def __init__(self, size: int, max_size: int):
        super().__init__(
            message=f"Payload too large: {size} bytes (max: {max_size} bytes)",
            error_code=ErrorCode.PAYLOAD_TOO_LARGE,
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.CLIENT_ERROR,
            details={"size": size, "max_size": max_size},
            suggestions=["Reduce payload size", "Split large requests"],
            http_status=413
        )


# GLM API Specific Exceptions

class GLMAPIError(BaseGLMException):
    """GLM API error."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(
            message=f"GLM API error: {message}",
            error_code=ErrorCode.GLM_API_ERROR,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.EXTERNAL_API_ERROR,
            details={"status_code": status_code, "response_data": response_data},
            suggestions=["Check GLM API status", "Verify API credentials", "Try again later"],
            http_status=502
        )


class GLMTimeoutError(BaseGLMException):
    """GLM API timeout error."""
    
    def __init__(self, timeout: int):
        super().__init__(
            message=f"GLM API request timed out after {timeout} seconds",
            error_code=ErrorCode.GLM_TIMEOUT,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.EXTERNAL_API_ERROR,
            details={"timeout": timeout},
            suggestions=["Increase timeout", "Simplify request", "Try again later"],
            http_status=504
        )


class GLMRateLimitError(BaseGLMException):
    """GLM API rate limit error."""
    
    def __init__(self, message: str = "GLM API rate limit exceeded"):
        super().__init__(
            message=message,
            error_code=ErrorCode.GLM_RATE_LIMIT,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.RATE_LIMIT_ERROR,
            suggestions=["Implement token rotation", "Reduce request frequency"],
            http_status=429
        )


class GLMInvalidModelError(BaseGLMException):
    """GLM invalid model error."""
    
    def __init__(self, model: str, available_models: List[str]):
        super().__init__(
            message=f"Invalid model: {model}",
            error_code=ErrorCode.GLM_INVALID_MODEL,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.CLIENT_ERROR,
            details={"requested_model": model, "available_models": available_models},
            suggestions=[f"Use one of: {', '.join(available_models)}"],
            http_status=400
        )


class GLMQuotaExceededError(BaseGLMException):
    """GLM quota exceeded error."""
    
    def __init__(self, message: str = "GLM API quota exceeded"):
        super().__init__(
            message=message,
            error_code=ErrorCode.GLM_QUOTA_EXCEEDED,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.EXTERNAL_API_ERROR,
            suggestions=["Check quota limits", "Upgrade subscription", "Wait for quota reset"],
            http_status=429
        )


# Token Management Exceptions

class TokenExhaustionError(BaseGLMException):
    """Token exhaustion error."""
    
    def __init__(self, message: str = "All API tokens exhausted"):
        super().__init__(
            message=message,
            error_code=ErrorCode.TOKEN_EXHAUSTION,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.CONFIGURATION_ERROR,
            suggestions=["Add more tokens", "Check token health", "Implement token recovery"],
            http_status=503
        )


class TokenInvalidError(BaseGLMException):
    """Token invalid error."""
    
    def __init__(self, token_id: str):
        super().__init__(
            message=f"Invalid token: {token_id}",
            error_code=ErrorCode.TOKEN_INVALID,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.CONFIGURATION_ERROR,
            details={"token_id": token_id},
            suggestions=["Check token format", "Verify token credentials"],
            http_status=401
        )


class TokenExpiredError(BaseGLMException):
    """Token expired error."""
    
    def __init__(self, token_id: str):
        super().__init__(
            message=f"Token expired: {token_id}",
            error_code=ErrorCode.TOKEN_EXPIRED,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.CONFIGURATION_ERROR,
            details={"token_id": token_id},
            suggestions=["Refresh token", "Update token configuration"],
            http_status=401
        )


# Configuration Exceptions

class ConfigurationError(BaseGLMException):
    """Configuration error."""
    
    def __init__(self, message: str, config_section: Optional[str] = None):
        super().__init__(
            message=f"Configuration error: {message}",
            error_code=ErrorCode.CONFIG_ERROR,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.CONFIGURATION_ERROR,
            details={"config_section": config_section},
            suggestions=["Check configuration file", "Verify environment variables"],
            http_status=500
        )


class ConfigValidationError(BaseGLMException):
    """Configuration validation error."""
    
    def __init__(self, message: str, validation_errors: Optional[List[str]] = None):
        super().__init__(
            message=f"Configuration validation error: {message}",
            error_code=ErrorCode.CONFIG_VALIDATION_ERROR,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.CONFIGURATION_ERROR,
            details={"validation_errors": validation_errors},
            suggestions=["Fix configuration values", "Check required fields"],
            http_status=500
        )


# Processing Exceptions

class ResponseProcessingError(BaseGLMException):
    """Response processing error."""
    
    def __init__(self, message: str, processing_step: str):
        super().__init__(
            message=f"Response processing error in {processing_step}: {message}",
            error_code=ErrorCode.RESPONSE_PROCESSING_ERROR,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.SERVER_ERROR,
            details={"processing_step": processing_step},
            suggestions=["Check response format", "Verify processing pipeline"],
            http_status=500
        )


class ContentFilteringError(BaseGLMException):
    """Content filtering error."""
    
    def __init__(self, message: str = "Content blocked by filter"):
        super().__init__(
            message=message,
            error_code=ErrorCode.CONTENT_FILTERING_ERROR,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.CLIENT_ERROR,
            suggestions=["Modify content", "Check content guidelines"],
            http_status=400
        )


class ThinkingTagProcessingError(BaseGLMException):
    """Thinking tag processing error."""
    
    def __init__(self, message: str):
        super().__init__(
            message=f"Thinking tag processing error: {message}",
            error_code=ErrorCode.THINKING_TAG_PROCESSING_ERROR,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.SERVER_ERROR,
            suggestions=["Check thinking tag format", "Verify processing logic"],
            http_status=500
        )


# Error Handler Registry

class ErrorHandler:
    """Central error handler for the application."""
    
    @staticmethod
    def handle_exception(e: Exception) -> Response:
        """Handle any exception and return appropriate response."""
        if isinstance(e, BaseGLMException):
            return ErrorHandler._handle_glm_exception(e)
        else:
            return ErrorHandler._handle_generic_exception(e)
    
    @staticmethod
    def _handle_glm_exception(e: BaseGLMException) -> Response:
        """Handle GLM-specific exceptions."""
        error_response = e.to_error_response()
        
        # Log the error (will be implemented with logging system)
        # logger.error(f"GLM Exception: {e.error_code.value}", extra=error_response.to_dict())
        
        return jsonify(error_response.to_dict()), e.http_status
    
    @staticmethod
    def _handle_generic_exception(e: Exception) -> Response:
        """Handle generic exceptions."""
        # Create a generic server error
        glm_exception = BaseGLMException(
            message="An unexpected error occurred",
            error_code=ErrorCode.INTERNAL_SERVER_ERROR,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.SERVER_ERROR,
            details={"original_error": str(e), "traceback": traceback.format_exc()},
            suggestions=["Try again later", "Contact support if problem persists"]
        )
        
        return ErrorHandler._handle_glm_exception(glm_exception)


# Flask error handlers

def register_error_handlers(app):
    """Register error handlers with Flask app."""
    
    @app.errorhandler(BaseGLMException)
    def handle_glm_exception(e):
        return ErrorHandler.handle_exception(e)
    
    @app.errorhandler(404)
    def handle_not_found(e):
        exception = ResourceNotFoundError("Endpoint", request.path)
        return ErrorHandler.handle_exception(exception)
    
    @app.errorhandler(405)
    def handle_method_not_allowed(e):
        exception = BaseGLMException(
            message=f"Method {request.method} not allowed for {request.path}",
            error_code=ErrorCode.METHOD_NOT_ALLOWED,
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.CLIENT_ERROR,
            http_status=405
        )
        return ErrorHandler.handle_exception(exception)
    
    @app.errorhandler(500)
    def handle_internal_error(e):
        return ErrorHandler.handle_exception(e)
    
    @app.errorhandler(Exception)
    def handle_generic_exception(e):
        return ErrorHandler.handle_exception(e)