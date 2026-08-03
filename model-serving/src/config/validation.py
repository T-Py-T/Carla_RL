"""
Configuration validation and error reporting system.

Provides comprehensive validation for configuration values with
detailed error reporting and validation results.
"""

import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from pydantic import ValidationError as PydanticValidationError

from .settings import BaseConfig, AppConfig


class ValidationError(Exception):
    """Custom configuration-validation error.

    Wraps a :class:`ValidationResult` so that callers can raise a single
    exception carrying the full diagnostic detail (errors, warnings, infos)
    surfaced by :func:`validate_config`. This lives alongside Pydantic's
    own ``ValidationError`` (imported as ``PydanticValidationError``) which
    is the low-level exception raised by model parsing; the two serve
    different layers of the configuration stack.
    """

    def __init__(self, message: str, validation_result: Optional["ValidationResult"] = None):
        super().__init__(message)
        self.validation_result = validation_result


class ValidationSeverity(str, Enum):
    """Validation error severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Individual validation issue."""
    field: str
    message: str
    severity: ValidationSeverity
    value: Any = None
    expected: Any = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Complete validation result."""
    is_valid: bool
    issues: List[ValidationIssue]
    warnings: List[ValidationIssue]
    errors: List[ValidationIssue]
    summary: Dict[str, int]

    def __post_init__(self):
        """Categorize issues by severity."""
        self.warnings = [i for i in self.issues if i.severity == ValidationSeverity.WARNING]
        self.errors = [i for i in self.issues if i.severity == ValidationSeverity.ERROR]
        self.is_valid = len(self.errors) == 0
        
        # Create summary
        self.summary = {
            "total": len(self.issues),
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "info": len([i for i in self.issues if i.severity == ValidationSeverity.INFO])
        }




class ConfigValidator:
    """Configuration validator with comprehensive validation rules."""

    def __init__(self):
        """Initialize configuration validator."""
        self._validators: Dict[str, List[Callable]] = {}
        self._custom_rules: List[Callable] = []
    
    def validate(self, config: BaseConfig) -> ValidationResult:
        """
        Validate configuration object.
        
        Args:
            config: Configuration object to validate
            
        Returns:
            Validation result with issues and summary
        """
        config_data = config.model_dump()
        issues = self._pydantic_issues(config, config_data)
        issues.extend(self._field_validation_issues(config, config_data))
        issues.extend(self._global_validation_issues(config))

        return ValidationResult(
            is_valid=not any(
                issue.severity == ValidationSeverity.ERROR for issue in issues
            ),
            issues=issues,
            warnings=[],
            errors=[],
            summary={},
        )

    @staticmethod
    def _pydantic_issues(
        config: BaseConfig, config_data: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Return issues emitted by Pydantic's model validation."""
        try:
            config.model_validate(config_data)
            return []
        except PydanticValidationError as exc:
            return [
                ValidationIssue(
                    field=".".join(str(part) for part in error["loc"]),
                    message=error["msg"],
                    severity=ValidationSeverity.ERROR,
                    value=error.get("input"),
                    expected=error.get("ctx", {}).get("expected"),
                )
                for error in exc.errors()
            ]

    def _field_validation_issues(
        self, config: BaseConfig, config_data: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Run registered field validators."""
        issues: List[ValidationIssue] = []
        for field_name, validators in self._validators.items():
            field_value = self._get_nested_value(config_data, field_name)
            for validator_func in validators:
                try:
                    self._append_result(
                        issues, validator_func(field_value, config)
                    )
                except Exception as exc:
                    issues.append(
                        ValidationIssue(
                            field=field_name,
                            message=f"Validation error: {exc}",
                            severity=ValidationSeverity.ERROR,
                            value=field_value,
                        )
                    )
        return issues

    def _global_validation_issues(self, config: BaseConfig) -> List[ValidationIssue]:
        """Run registered whole-configuration rules."""
        issues: List[ValidationIssue] = []
        for rule in self._custom_rules:
            try:
                self._append_result(issues, rule(config))
            except Exception as exc:
                issues.append(
                    ValidationIssue(
                        field="global",
                        message=f"Global validation error: {exc}",
                        severity=ValidationSeverity.ERROR,
                    )
                )
        return issues

    @staticmethod
    def _append_result(issues: List[ValidationIssue], result: Any) -> None:
        if isinstance(result, ValidationIssue):
            issues.append(result)
        elif isinstance(result, list):
            issues.extend(result)
    
    def add_field_validator(self, field_name: str, validator_func: Callable) -> None:
        """
        Add field-specific validator.
        
        Args:
            field_name: Field name (supports dot notation for nested fields)
            validator_func: Validator function
        """
        if field_name not in self._validators:
            self._validators[field_name] = []
        self._validators[field_name].append(validator_func)
    
    def add_global_validator(self, validator_func: Callable) -> None:
        """
        Add global validator.
        
        Args:
            validator_func: Global validator function
        """
        self._custom_rules.append(validator_func)
    
    def _get_nested_value(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get nested value from dictionary using dot notation."""
        keys = field_path.split(".")
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current


# Built-in validators
def validate_port(value: Any, config: BaseConfig) -> Optional[ValidationIssue]:
    """Validate port number."""
    if not isinstance(value, int) or not (1 <= value <= 65535):
        return ValidationIssue(
            field="port",
            message=f"Port must be an integer between 1 and 65535, got {value}",
            severity=ValidationSeverity.ERROR,
            value=value,
            expected="1-65535"
        )
    return None


def validate_host(value: Any, config: BaseConfig) -> Optional[ValidationIssue]:
    """Validate host address."""
    if not isinstance(value, str):
        return ValidationIssue(
            field="host",
            message=f"Host must be a string, got {type(value).__name__}",
            severity=ValidationSeverity.ERROR,
            value=value
        )
    
    # Basic IP/hostname validation
    if value not in ["0.0.0.0", "127.0.0.1", "localhost"]:
        # Simple hostname validation
        if not re.match(r'^[a-zA-Z0-9.-]+$', value):
            return ValidationIssue(
                field="host",
                message=f"Invalid host format: {value}",
                severity=ValidationSeverity.WARNING,
                value=value,
                suggestion="Use a valid IP address or hostname"
            )
    
    return None


def validate_file_path(value: Any, config: BaseConfig) -> Optional[ValidationIssue]:
    """Validate file path exists."""
    if value is None:
        return None
    
    path = Path(value)
    if not path.exists():
        return ValidationIssue(
            field="file_path",
            message=f"File does not exist: {value}",
            severity=ValidationSeverity.ERROR,
            value=value,
            suggestion="Ensure the file exists or provide a valid path"
        )
    
    if not path.is_file():
        return ValidationIssue(
            field="file_path",
            message=f"Path is not a file: {value}",
            severity=ValidationSeverity.ERROR,
            value=value,
            suggestion="Provide a path to a file, not a directory"
        )
    
    return None


def validate_directory_path(value: Any, config: BaseConfig) -> Optional[ValidationIssue]:
    """Validate directory path exists."""
    if value is None:
        return None
    
    path = Path(value)
    if not path.exists():
        return ValidationIssue(
            field="directory_path",
            message=f"Directory does not exist: {value}",
            severity=ValidationSeverity.WARNING,
            value=value,
            suggestion="Directory will be created automatically"
        )
    elif not path.is_dir():
        return ValidationIssue(
            field="directory_path",
            message=f"Path is not a directory: {value}",
            severity=ValidationSeverity.ERROR,
            value=value,
            suggestion="Provide a path to a directory"
        )
    
    return None


def validate_url(value: Any, config: BaseConfig) -> Optional[ValidationIssue]:
    """Validate URL format."""
    if value is None:
        return None
    
    if not isinstance(value, str):
        return ValidationIssue(
            field="url",
            message=f"URL must be a string, got {type(value).__name__}",
            severity=ValidationSeverity.ERROR,
            value=value
        )
    
    # Basic URL validation
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    if not url_pattern.match(value):
        return ValidationIssue(
            field="url",
            message=f"Invalid URL format: {value}",
            severity=ValidationSeverity.ERROR,
            value=value,
            suggestion="Use a valid URL format (e.g., http://example.com:8080)"
        )
    
    return None


def validate_environment_consistency(config: AppConfig) -> List[ValidationIssue]:
    """Validate environment-specific configuration consistency."""
    issues = []
    
    # Handle both enum objects and strings
    environment = config.environment.value if hasattr(config.environment, 'value') else str(config.environment)
    
    if environment == "production":
        # Production-specific validations
        if config.debug:
            issues.append(ValidationIssue(
                field="debug",
                message="Debug mode should not be enabled in production",
                severity=ValidationSeverity.ERROR,
                value=config.debug,
                suggestion="Set debug=False for production"
            ))
        
        if config.server.reload:
            issues.append(ValidationIssue(
                field="server.reload",
                message="Auto-reload should not be enabled in production",
                severity=ValidationSeverity.ERROR,
                value=config.server.reload,
                suggestion="Set reload=False for production"
            ))
        
        if not config.security.enabled:
            issues.append(ValidationIssue(
                field="security.enabled",
                message="Security should be enabled in production",
                severity=ValidationSeverity.WARNING,
                value=config.security.enabled,
                suggestion="Enable security features for production"
            ))
        
        if config.server.workers == 1:
            issues.append(ValidationIssue(
                field="server.workers",
                message="Consider using multiple workers in production",
                severity=ValidationSeverity.INFO,
                value=config.server.workers,
                suggestion="Set workers > 1 for better performance"
            ))
    
    return issues


def validate_database_config(config: AppConfig) -> List[ValidationIssue]:
    """Validate database configuration."""
    issues = []
    db_config = config.database
    
    # Handle both enum objects and strings
    backend = db_config.backend.value if hasattr(db_config.backend, 'value') else str(db_config.backend)
    
    if backend in ["postgresql", "mysql"]:
        if not db_config.host and not db_config.url:
            issues.append(ValidationIssue(
                field="database.host",
                message=f"Host or URL required for {backend}",
                severity=ValidationSeverity.ERROR,
                suggestion="Provide database host or connection URL"
            ))
        
        if not db_config.username:
            issues.append(ValidationIssue(
                field="database.username",
                message=f"Username required for {backend}",
                severity=ValidationSeverity.ERROR,
                suggestion="Provide database username"
            ))
    
    return issues


def validate_cache_config(config: AppConfig) -> List[ValidationIssue]:
    """Validate cache configuration."""
    issues = []
    cache_config = config.cache
    
    # Handle both enum objects and strings
    backend = cache_config.backend.value if hasattr(cache_config.backend, 'value') else str(cache_config.backend)
    
    if backend in ["redis", "memcached"] and not cache_config.host:
        issues.append(ValidationIssue(
            field="cache.host",
            message=f"Host required for {backend}",
            severity=ValidationSeverity.ERROR,
            suggestion="Provide cache host address"
        ))
    
    return issues


def validate_model_config(config: AppConfig) -> List[ValidationIssue]:
    """Validate model configuration."""
    issues = []
    model_config = config.model
    
    if not model_config.model_path.exists():
        issues.append(ValidationIssue(
            field="model.model_path",
            message=f"Model path does not exist: {model_config.model_path}",
            severity=ValidationSeverity.ERROR,
            value=str(model_config.model_path),
            suggestion="Provide a valid path to the model directory"
        ))
    
    if model_config.model_file and not model_config.model_file.exists():
        issues.append(ValidationIssue(
            field="model.model_file",
            message=f"Model file does not exist: {model_config.model_file}",
            severity=ValidationSeverity.ERROR,
            value=str(model_config.model_file),
            suggestion="Provide a valid path to the model file"
        ))
    
    if model_config.max_batch_size < model_config.batch_size:
        issues.append(ValidationIssue(
            field="model.max_batch_size",
            message="max_batch_size must be >= batch_size",
            severity=ValidationSeverity.ERROR,
            value=model_config.max_batch_size,
            expected=f">= {model_config.batch_size}"
        ))
    
    return issues


def create_default_validator() -> ConfigValidator:
    """Create validator with default validation rules."""
    validator = ConfigValidator()
    
    # Add field validators
    validator.add_field_validator("server.host", validate_host)
    validator.add_field_validator("server.port", validate_port)
    validator.add_field_validator("model.model_path", validate_directory_path)
    validator.add_field_validator("model.model_file", validate_file_path)
    validator.add_field_validator("logging.file_path", validate_file_path)
    validator.add_field_validator("database.url", validate_url)
    validator.add_field_validator("monitoring.jaeger_endpoint", validate_url)
    
    # Add global validators
    validator.add_global_validator(validate_environment_consistency)
    validator.add_global_validator(validate_database_config)
    validator.add_global_validator(validate_cache_config)
    validator.add_global_validator(validate_model_config)
    
    return validator


def validate_config(config: BaseConfig, validator: Optional[ConfigValidator] = None) -> ValidationResult:
    """
    Validate configuration with optional custom validator.
    
    Args:
        config: Configuration to validate
        validator: Optional custom validator
        
    Returns:
        Validation result
    """
    if validator is None:
        validator = create_default_validator()
    
    return validator.validate(config)


def format_validation_result(result: ValidationResult) -> str:
    """
    Format validation result as human-readable string.
    
    Args:
        result: Validation result to format
        
    Returns:
        Formatted string
    """
    lines = []
    
    # Summary
    lines.append("Configuration Validation Result")
    lines.append("=" * 40)
    lines.append(f"Valid: {result.is_valid}")
    lines.append(f"Total Issues: {result.summary['total']}")
    lines.append(f"Errors: {result.summary['errors']}")
    lines.append(f"Warnings: {result.summary['warnings']}")
    lines.append(f"Info: {result.summary['info']}")
    lines.append("")

    for severity in [ValidationSeverity.ERROR, ValidationSeverity.WARNING, ValidationSeverity.INFO]:
        lines.extend(_format_severity_issues(result.issues, severity))

    return "\n".join(lines)


def _format_severity_issues(
    all_issues: List[ValidationIssue], severity: ValidationSeverity
) -> List[str]:
    """Format all issues for one severity group."""
    issues = [issue for issue in all_issues if issue.severity == severity]
    if not issues:
        return []
    lines = [f"{severity.value.upper()}S:", "-" * 20]
    for issue in issues:
        lines.extend(_format_issue(issue))
    return lines


def _format_issue(issue: ValidationIssue) -> List[str]:
    """Format one validation issue."""
    lines = [f"Field: {issue.field}", f"Message: {issue.message}"]
    optional_values = (
        ("Value", issue.value),
        ("Expected", issue.expected),
        ("Suggestion", issue.suggestion),
    )
    lines.extend(f"{label}: {value}" for label, value in optional_values if value is not None)
    lines.append("")
    return lines
