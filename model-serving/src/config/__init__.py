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

from .settings import (
    BaseConfig, AppConfig, ServerConfig, ModelConfig, LoggingConfig,
    MonitoringConfig, DatabaseConfig, CacheConfig, SecurityConfig,
    Environment, LogLevel, ModelBackend, DatabaseBackend, CacheBackend
)
from .loader import ConfigLoader, load_config
from .validation import (
    ValidationSeverity, ValidationIssue, ValidationResult, ValidationError,
    ConfigValidator, validate_config, format_validation_result
)

__all__ = [
    # Settings
    "BaseConfig", "AppConfig", "ServerConfig", "ModelConfig", "LoggingConfig",
    "MonitoringConfig", "DatabaseConfig", "CacheConfig", "SecurityConfig",
    "Environment", "LogLevel", "ModelBackend", "DatabaseBackend", "CacheBackend",
    
    # Loader
    "ConfigLoader", "load_config",
    
    # Validation
    "ValidationSeverity", "ValidationIssue", "ValidationResult", "ValidationError",
    "ConfigValidator", "validate_config", "format_validation_result",
]