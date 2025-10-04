"""
Configuration Management System for GLM OpenAI-Compatible API Wrapper

This module provides a comprehensive configuration system with validation,
environment variable support, and type safety using Pydantic.
"""

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator, root_validator
from pydantic_settings import BaseSettings


class LogLevel(str, Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Log format enumeration."""
    JSON = "json"
    TEXT = "text"


class TokenRotationStrategy(str, Enum):
    """Token rotation strategy enumeration."""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_USED = "least_used"
    HEALTH_BASED = "health_based"


class Environment(str, Enum):
    """Environment enumeration."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class APIConfig(BaseSettings):
    """API server configuration."""
    
    host: str = Field(default="0.0.0.0", description="Server host address")
    port: int = Field(default=8976, ge=1, le=65535, description="Server port")
    debug: bool = Field(default=False, description="Enable debug mode")
    workers: int = Field(default=4, ge=1, le=32, description="Number of worker processes")
    reload: bool = Field(default=False, description="Enable auto-reload in development")
    access_log: bool = Field(default=True, description="Enable access logging")
    
    class Config:
        env_prefix = "API_"


class GLMConfig(BaseSettings):
    """GLM API configuration."""
    
    base_url: str = Field(
        default="https://chat.z.ai/api/chat/completions",
        description="GLM API base URL"
    )
    timeout: int = Field(default=120, ge=1, le=600, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, ge=0.1, le=60.0, description="Retry delay in seconds")
    connection_pool_size: int = Field(default=20, ge=1, le=100, description="HTTP connection pool size")
    max_pool_connections: int = Field(default=20, ge=1, le=100, description="Maximum pool connections")
    
    # Model configuration
    default_model: str = Field(default="0727-360B-API", description="Default GLM model")
    available_models: List[str] = Field(
        default=["0727-360B-API", "glm-4", "glm-3-turbo"],
        description="List of available models"
    )
    
    # Rate limiting
    requests_per_minute: int = Field(default=60, ge=1, description="Requests per minute limit")
    burst_limit: int = Field(default=10, ge=1, description="Burst request limit")
    
    @validator('available_models')
    def validate_models(cls, v):
        if not v:
            raise ValueError("At least one model must be available")
        return v
    
    @validator('default_model')
    def validate_default_model(cls, v, values):
        if 'available_models' in values and v not in values['available_models']:
            raise ValueError(f"Default model '{v}' must be in available_models list")
        return v
    
    class Config:
        env_prefix = "GLM_"


class TokenConfig(BaseSettings):
    """Token management configuration."""
    
    token_file: str = Field(default="token.json", description="Token file path")
    rotation_strategy: TokenRotationStrategy = Field(
        default=TokenRotationStrategy.HEALTH_BASED,
        description="Token rotation strategy"
    )
    health_check_interval: int = Field(
        default=300, ge=60, description="Health check interval in seconds"
    )
    error_threshold: int = Field(
        default=5, ge=1, description="Error threshold before marking token as degraded"
    )
    recovery_attempts: int = Field(
        default=3, ge=1, description="Recovery attempts before blacklisting token"
    )
    token_timeout: int = Field(
        default=30, ge=5, description="Individual token request timeout"
    )
    
    # Token usage tracking
    track_usage: bool = Field(default=True, description="Enable token usage tracking")
    usage_window: int = Field(default=3600, ge=300, description="Usage tracking window in seconds")
    cost_tracking: bool = Field(default=True, description="Enable cost tracking")
    
    class Config:
        env_prefix = "TOKEN_"


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    
    level: LogLevel = Field(default=LogLevel.INFO, description="Log level")
    format: LogFormat = Field(default=LogFormat.JSON, description="Log format")
    file_path: Optional[str] = Field(default="logs/glm_server.log", description="Log file path")
    max_size: str = Field(default="100MB", description="Maximum log file size")
    backup_count: int = Field(default=5, ge=1, description="Number of backup log files")
    
    # Structured logging
    enable_correlation_id: bool = Field(default=True, description="Enable correlation ID logging")
    enable_request_logging: bool = Field(default=True, description="Enable request logging")
    enable_response_logging: bool = Field(default=True, description="Enable response logging")
    
    # Performance logging
    log_slow_requests: bool = Field(default=True, description="Log slow requests")
    slow_request_threshold: float = Field(
        default=5.0, ge=0.1, description="Slow request threshold in seconds"
    )
    
    class Config:
        env_prefix = "LOG_"


class SecurityConfig(BaseSettings):
    """Security configuration."""
    
    # API Key authentication
    require_api_key: bool = Field(default=False, description="Require API key authentication")
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    api_keys: List[str] = Field(default=[], description="Valid API keys")
    
    # CORS configuration
    enable_cors: bool = Field(default=True, description="Enable CORS")
    cors_origins: List[str] = Field(default=["*"], description="Allowed CORS origins")
    cors_methods: List[str] = Field(
        default=["GET", "POST", "OPTIONS"], description="Allowed CORS methods"
    )
    
    # Rate limiting
    enable_rate_limiting: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_per_minute: int = Field(default=60, ge=1, description="Rate limit per minute")
    rate_limit_per_hour: int = Field(default=1000, ge=1, description="Rate limit per hour")
    
    # Content filtering
    enable_content_filtering: bool = Field(default=True, description="Enable content filtering")
    max_content_length: int = Field(
        default=100000, ge=1000, description="Maximum content length in characters"
    )
    
    class Config:
        env_prefix = "SECURITY_"


class MonitoringConfig(BaseSettings):
    """Monitoring and observability configuration."""
    
    # Metrics
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(default=8977, ge=1, le=65535, description="Metrics server port")
    metrics_path: str = Field(default="/metrics", description="Metrics endpoint path")
    
    # Health checks
    enable_health_checks: bool = Field(default=True, description="Enable health checks")
    health_check_path: str = Field(default="/health", description="Health check endpoint path")
    health_check_timeout: int = Field(
        default=10, ge=1, description="Health check timeout in seconds"
    )
    
    # Tracing
    enable_tracing: bool = Field(default=False, description="Enable distributed tracing")
    tracing_sample_rate: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Tracing sample rate"
    )
    
    class Config:
        env_prefix = "MONITORING_"


class CacheConfig(BaseSettings):
    """Cache configuration."""
    
    # Redis configuration
    enable_cache: bool = Field(default=False, description="Enable Redis cache")
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis URL")
    redis_timeout: int = Field(default=5, ge=1, description="Redis timeout in seconds")
    
    # Cache TTL settings
    conversation_ttl: int = Field(default=3600, ge=60, description="Conversation cache TTL")
    model_info_ttl: int = Field(default=300, ge=60, description="Model info cache TTL")
    token_stats_ttl: int = Field(default=60, ge=10, description="Token stats cache TTL")
    
    class Config:
        env_prefix = "CACHE_"


class ApplicationConfig(BaseSettings):
    """Main application configuration that combines all config sections."""
    
    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Application environment")
    app_name: str = Field(default="GLM-OpenAI-Wrapper", description="Application name")
    app_version: str = Field(default="2.0.0", description="Application version")
    
    # Configuration sections
    api: APIConfig = Field(default_factory=APIConfig)
    glm: GLMConfig = Field(default_factory=GLMConfig)
    token: TokenConfig = Field(default_factory=TokenConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    
    @root_validator
    def validate_environment_config(cls, values):
        """Validate configuration based on environment."""
        env = values.get('environment')
        
        if env == Environment.PRODUCTION:
            # Production-specific validations
            if values.get('api', {}).debug:
                raise ValueError("Debug mode should be disabled in production")
            if values.get('logging', {}).level == LogLevel.DEBUG:
                raise ValueError("Debug logging should be disabled in production")
        
        return values
    
    class Config:
        env_prefix = "APP_"
        case_sensitive = False
        
    @classmethod
    def load_from_env(cls) -> 'ApplicationConfig':
        """Load configuration from environment variables and .env file."""
        from dotenv import load_dotenv
        
        # Load .env file if it exists
        env_file = Path(".env")
        if env_file.exists():
            load_dotenv(env_file)
        
        return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.dict()
    
    def get_log_level(self) -> str:
        """Get the configured log level as string."""
        return self.logging.level.value
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION


# Global configuration instance
config: Optional[ApplicationConfig] = None


def get_config() -> ApplicationConfig:
    """Get the global configuration instance."""
    global config
    if config is None:
        config = ApplicationConfig.load_from_env()
    return config


def reload_config() -> ApplicationConfig:
    """Reload configuration from environment."""
    global config
    config = ApplicationConfig.load_from_env()
    return config


def validate_config() -> bool:
    """Validate the current configuration."""
    try:
        get_config()
        return True
    except Exception:
        return False