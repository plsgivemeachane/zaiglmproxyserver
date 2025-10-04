"""
Structured Logging System for GLM OpenAI-Compatible API Wrapper

This module provides a comprehensive logging framework with correlation IDs,
structured formatting, performance monitoring, and contextual logging.
"""

import contextvars
import functools
import json
import logging
import logging.handlers
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union
from uuid import uuid4

import structlog
from flask import g, request
from pythonjsonlogger import jsonlogger

from config.settings import LoggingConfig, LogLevel, LogFormat


# Context variable for correlation ID
correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar('correlation_id', default='')

# Context variable for user ID
user_id: contextvars.ContextVar[str] = contextvars.ContextVar('user_id', default='')

# Context variable for request context
request_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar('request_context', default={})


class CorrelationIDProcessor:
    """Processor to add correlation ID to log entries."""
    
    def __call__(self, logger, method_name, event_dict):
        """Add correlation ID to event dict."""
        corr_id = correlation_id.get('')
        if corr_id:
            event_dict['correlation_id'] = corr_id
        
        user = user_id.get('')
        if user:
            event_dict['user_id'] = user
        
        # Add request context if available
        ctx = request_context.get({})
        if ctx:
            event_dict.update(ctx)
        
        return event_dict


class TimestampProcessor:
    """Processor to add timestamp to log entries."""
    
    def __call__(self, logger, method_name, event_dict):
        """Add timestamp to event dict."""
        event_dict['timestamp'] = datetime.utcnow().isoformat()
        return event_dict


class ErrorProcessor:
    """Processor to format exception information."""
    
    def __call__(self, logger, method_name, event_dict):
        """Format exception information."""
        if 'exc_info' in event_dict:
            exc_info = event_dict.pop('exc_info')
            if exc_info:
                event_dict['exception'] = {
                    'type': exc_info[0].__name__ if exc_info[0] else None,
                    'message': str(exc_info[1]) if exc_info[1] else None,
                    'traceback': traceback.format_exception(*exc_info) if exc_info[0] else None
                }
        return event_dict


class PerformanceProcessor:
    """Processor to add performance metrics."""
    
    def __call__(self, logger, method_name, event_dict):
        """Add performance information if available."""
        if hasattr(g, 'request_start_time'):
            duration = time.time() - g.request_start_time
            event_dict['request_duration'] = round(duration, 4)
        
        return event_dict


class CustomJSONFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields."""
    
    def add_fields(self, log_record, record, message_dict):
        """Add custom fields to log record."""
        super().add_fields(log_record, record, message_dict)
        
        # Add standard fields
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno
        
        # Add process information
        log_record['process_id'] = record.process
        log_record['thread_id'] = record.thread
        
        # Add timestamp if not present
        if 'timestamp' not in log_record:
            log_record['timestamp'] = datetime.utcnow().isoformat()


class LoggingManager:
    """Central logging manager for the application."""
    
    def __init__(self, config: LoggingConfig):
        self.config = config
        self.logger = None
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        # Configure structlog
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="ISO"),
            CorrelationIDProcessor(),
            TimestampProcessor(),
            ErrorProcessor(),
            PerformanceProcessor(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]
        
        if self.config.format == LogFormat.JSON:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.processors.KeyValueRenderer())
        
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        # Configure standard library logging
        self._setup_stdlib_logging()
        
        # Get structured logger
        self.logger = structlog.get_logger("glm_server")
    
    def _setup_stdlib_logging(self):
        """Setup standard library logging configuration."""
        # Create root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.config.level.value)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create formatters
        if self.config.format == LogFormat.JSON:
            formatter = CustomJSONFormatter(
                '%(timestamp)s %(level)s %(name)s %(message)s'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(correlation_id)s - %(message)s'
            )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(self.config.level.value)
        root_logger.addHandler(console_handler)
        
        # File handler (if configured)
        if self.config.file_path:
            file_path = Path(self.config.file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Parse max_size
            max_bytes = self._parse_size(self.config.max_size)
            
            file_handler = logging.handlers.RotatingFileHandler(
                filename=file_path,
                maxBytes=max_bytes,
                backupCount=self.config.backup_count,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(self.config.level.value)
            root_logger.addHandler(file_handler)
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string (e.g., '100MB') to bytes."""
        size_str = size_str.upper()
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    def get_logger(self, name: str = None) -> structlog.BoundLogger:
        """Get a structured logger instance."""
        if name:
            return structlog.get_logger(name)
        return self.logger


class RequestLogger:
    """Logger for HTTP requests and responses."""
    
    def __init__(self, logger: structlog.BoundLogger, config: LoggingConfig):
        self.logger = logger
        self.config = config
    
    def log_request_start(self, request_data: Dict[str, Any]):
        """Log the start of a request."""
        if not self.config.enable_request_logging:
            return
        
        # Set request start time
        g.request_start_time = time.time()
        
        # Generate correlation ID if not present
        corr_id = request.headers.get('X-Correlation-ID', str(uuid4()))
        correlation_id.set(corr_id)
        
        # Set request context
        context = {
            'request_method': request.method,
            'request_path': request.path,
            'request_remote_addr': request.remote_addr,
            'request_user_agent': request.headers.get('User-Agent', ''),
            'request_content_length': request.content_length
        }
        request_context.set(context)
        
        self.logger.info(
            "request_started",
            method=request.method,
            path=request.path,
            remote_addr=request.remote_addr,
            user_agent=request.headers.get('User-Agent', ''),
            content_length=request.content_length,
            **request_data
        )
    
    def log_request_end(self, response_data: Dict[str, Any], status_code: int):
        """Log the end of a request."""
        if not self.config.enable_response_logging:
            return
        
        duration = None
        if hasattr(g, 'request_start_time'):
            duration = time.time() - g.request_start_time
        
        log_data = {
            "status_code": status_code,
            "duration": round(duration, 4) if duration else None,
            **response_data
        }
        
        # Log as warning if slow request
        if (duration and self.config.log_slow_requests and 
            duration > self.config.slow_request_threshold):
            self.logger.warning("slow_request", **log_data)
        else:
            self.logger.info("request_completed", **log_data)
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log an error that occurred during request processing."""
        self.logger.error(
            "request_error",
            error_type=type(error).__name__,
            error_message=str(error),
            exc_info=True,
            **(context or {})
        )


class GLMAPILogger:
    """Logger for GLM API interactions."""
    
    def __init__(self, logger: structlog.BoundLogger):
        self.logger = logger
    
    def log_api_request(self, url: str, method: str, headers: Dict[str, str], 
                       payload: Dict[str, Any], token_id: str):
        """Log GLM API request."""
        # Sanitize headers (remove sensitive information)
        safe_headers = {k: v for k, v in headers.items() 
                       if k.lower() not in ['authorization', 'x-api-key']}
        
        self.logger.info(
            "glm_api_request",
            url=url,
            method=method,
            headers=safe_headers,
            payload_size=len(json.dumps(payload)) if payload else 0,
            token_id=token_id[:8] + "..." if token_id else None,
            model=payload.get('model') if payload else None,
            stream=payload.get('stream') if payload else None
        )
    
    def log_api_response(self, status_code: int, response_time: float, 
                        response_size: int, token_usage: Dict[str, Any] = None):
        """Log GLM API response."""
        self.logger.info(
            "glm_api_response",
            status_code=status_code,
            response_time=round(response_time, 4),
            response_size=response_size,
            token_usage=token_usage
        )
    
    def log_api_error(self, error: Exception, status_code: int = None, 
                     response_data: Dict[str, Any] = None):
        """Log GLM API error."""
        self.logger.error(
            "glm_api_error",
            error_type=type(error).__name__,
            error_message=str(error),
            status_code=status_code,
            response_data=response_data,
            exc_info=True
        )


class TokenLogger:
    """Logger for token management operations."""
    
    def __init__(self, logger: structlog.BoundLogger):
        self.logger = logger
    
    def log_token_rotation(self, old_token_id: str, new_token_id: str, reason: str):
        """Log token rotation."""
        self.logger.info(
            "token_rotation",
            old_token_id=old_token_id[:8] + "...",
            new_token_id=new_token_id[:8] + "...",
            reason=reason
        )
    
    def log_token_health_check(self, token_id: str, health_score: float, 
                              status: str, metrics: Dict[str, Any]):
        """Log token health check."""
        self.logger.info(
            "token_health_check",
            token_id=token_id[:8] + "...",
            health_score=health_score,
            status=status,
            metrics=metrics
        )
    
    def log_token_error(self, token_id: str, error_type: str, error_message: str):
        """Log token error."""
        self.logger.warning(
            "token_error",
            token_id=token_id[:8] + "...",
            error_type=error_type,
            error_message=error_message
        )


def log_function_call(logger: structlog.BoundLogger = None):
    """Decorator to log function calls with parameters and results."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = structlog.get_logger(func.__module__)
            
            start_time = time.time()
            
            # Log function entry
            logger.debug(
                "function_entry",
                function=func.__name__,
                module=func.__module__,
                args_count=len(args),
                kwargs_keys=list(kwargs.keys())
            )
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log successful completion
                logger.debug(
                    "function_exit",
                    function=func.__name__,
                    duration=round(duration, 4),
                    success=True
                )
                
                return result
            
            except Exception as e:
                duration = time.time() - start_time
                
                # Log error
                logger.error(
                    "function_error",
                    function=func.__name__,
                    duration=round(duration, 4),
                    error_type=type(e).__name__,
                    error_message=str(e),
                    exc_info=True
                )
                
                raise
        
        return wrapper
    return decorator


def set_correlation_id(corr_id: str):
    """Set correlation ID for current context."""
    correlation_id.set(corr_id)


def get_correlation_id() -> str:
    """Get current correlation ID."""
    return correlation_id.get('')


def set_user_id(user: str):
    """Set user ID for current context."""
    user_id.set(user)


def get_user_id() -> str:
    """Get current user ID."""
    return user_id.get('')


def set_request_context(context: Dict[str, Any]):
    """Set request context."""
    request_context.set(context)


def get_request_context() -> Dict[str, Any]:
    """Get current request context."""
    return request_context.get({})


# Global logging manager instance
_logging_manager: Optional[LoggingManager] = None


def initialize_logging(config: LoggingConfig) -> LoggingManager:
    """Initialize the logging system."""
    global _logging_manager
    _logging_manager = LoggingManager(config)
    return _logging_manager


def get_logger(name: str = None) -> structlog.BoundLogger:
    """Get a logger instance."""
    if _logging_manager is None:
        raise RuntimeError("Logging system not initialized. Call initialize_logging() first.")
    return _logging_manager.get_logger(name)


def get_request_logger() -> RequestLogger:
    """Get request logger instance."""
    logger = get_logger("request")
    from config.settings import get_config
    config = get_config()
    return RequestLogger(logger, config.logging)


def get_glm_api_logger() -> GLMAPILogger:
    """Get GLM API logger instance."""
    logger = get_logger("glm_api")
    return GLMAPILogger(logger)


def get_token_logger() -> TokenLogger:
    """Get token logger instance."""
    logger = get_logger("token")
    return TokenLogger(logger)


# Flask logging middleware
def setup_request_logging(app):
    """Setup request logging middleware for Flask app."""
    
    @app.before_request
    def before_request():
        """Log request start."""
        request_logger = get_request_logger()
        request_data = {
            'request_id': get_correlation_id() or str(uuid4())
        }
        
        # Set correlation ID if not present
        if not get_correlation_id():
            set_correlation_id(request_data['request_id'])
        
        request_logger.log_request_start(request_data)
    
    @app.after_request
    def after_request(response):
        """Log request completion."""
        request_logger = get_request_logger()
        response_data = {
            'response_size': response.content_length or 0
        }
        request_logger.log_request_end(response_data, response.status_code)
        return response