"""
Configuration management module.

Provides comprehensive configuration management including:
- Pydantic-based configuration models
- Hierarchical configuration loading
- Hot-reloading capabilities
- Validation and error reporting
- Environment-specific profiles
- Configuration templates
- Schema documentation
"""

from .diff import compare_configs
from .loader import ConfigLoader, load_config
from .schema import SchemaFormat, generate_schema_docs
from .settings import (
    AppConfig,
    BaseConfig,
    CacheBackend,
    CacheConfig,
    DatabaseBackend,
    DatabaseConfig,
    Environment,
    LoggingConfig,
    LogLevel,
    ModelBackend,
    ModelConfig,
    MonitoringConfig,
    SecurityConfig,
    ServerConfig,
)
from .templates import TemplateEngine
from .validation import (
    ConfigValidator,
    ValidationError,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    format_validation_result,
    validate_config,
)

__all__ = [
    # Settings
    "BaseConfig",
    "AppConfig",
    "ServerConfig",
    "ModelConfig",
    "LoggingConfig",
    "MonitoringConfig",
    "DatabaseConfig",
    "CacheConfig",
    "SecurityConfig",
    "Environment",
    "LogLevel",
    "ModelBackend",
    "DatabaseBackend",
    "CacheBackend",
    # Loader
    "ConfigLoader",
    "load_config",
    "compare_configs",
    "SchemaFormat",
    "generate_schema_docs",
    "TemplateEngine",
    # Validation
    "ValidationSeverity",
    "ValidationIssue",
    "ValidationResult",
    "ValidationError",
    "ConfigValidator",
    "validate_config",
    "format_validation_result",
]
