"""
Tests for configuration validation and error reporting system.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from src.config.validation import (
    ValidationSeverity, ValidationIssue, ValidationResult, ValidationError,
    ConfigValidator, validate_port, validate_host, validate_file_path,
    validate_directory_path, validate_url, validate_environment_consistency,
    validate_database_config, validate_cache_config, validate_model_config,
    create_default_validator, validate_config, format_validation_result
)
from src.config.settings import AppConfig, ServerConfig, ModelConfig, DatabaseConfig, CacheConfig, SecurityConfig, Environment, DatabaseBackend, CacheBackend


class TestValidationIssue:
    """Test ValidationIssue dataclass."""
    
    def test_validation_issue_creation(self):
        """Test creating validation issue."""
        issue = ValidationIssue(
            field="test.field",
            message="Test message",
            severity=ValidationSeverity.ERROR,
            value="test_value",
            expected="expected_value",
            suggestion="Test suggestion"
        )
        
        assert issue.field == "test.field"
        assert issue.message == "Test message"
        assert issue.severity == ValidationSeverity.ERROR
        assert issue.value == "test_value"
        assert issue.expected == "expected_value"
        assert issue.suggestion == "Test suggestion"


class TestValidationResult:
    """Test ValidationResult dataclass."""
    
    def test_validation_result_creation(self):
        """Test creating validation result."""
        issues = [
            ValidationIssue("field1", "Error message", ValidationSeverity.ERROR),
            ValidationIssue("field2", "Warning message", ValidationSeverity.WARNING),
            ValidationIssue("field3", "Info message", ValidationSeverity.INFO)
        ]
        
        result = ValidationResult(
            is_valid=False,
            issues=issues,
            warnings=[],
            errors=[],
            summary={}
        )
        
        assert result.is_valid is False
        assert len(result.issues) == 3
        assert len(result.warnings) == 1
        assert len(result.errors) == 1
        assert result.summary["total"] == 3
        assert result.summary["errors"] == 1
        assert result.summary["warnings"] == 1
        assert result.summary["info"] == 1


class TestConfigValidator:
    """Test ConfigValidator class."""
    
    def test_initialization(self):
        """Test validator initialization."""
        validator = ConfigValidator()
        assert validator._validators == {}
        assert validator._custom_rules == []
    
    def test_add_field_validator(self):
        """Test adding field validator."""
        validator = ConfigValidator()
        
        def test_validator(value, config):
            return None
        
        validator.add_field_validator("test.field", test_validator)
        assert "test.field" in validator._validators
        assert len(validator._validators["test.field"]) == 1
    
    def test_add_global_validator(self):
        """Test adding global validator."""
        validator = ConfigValidator()
        
        def test_validator(config):
            return []
        
        validator.add_global_validator(test_validator)
        assert len(validator._custom_rules) == 1
    
    def test_validate_with_pydantic_errors(self):
        """Test validation with Pydantic validation errors."""
        validator = ConfigValidator()
        
        # Create a mock config that will trigger Pydantic validation errors
        class MockConfig:
            def model_dump(self):
                return {"server": {"port": -1}}
            def model_validate(self, data):
                from pydantic import ValidationError
                from pydantic_core import ErrorDetails
                error_details = ErrorDetails(
                    type="greater_than_equal",
                    loc=("server", "port"),
                    msg="Input should be greater than or equal to 1",
                    input=-1,
                    ctx={"ge": 1}
                )
                raise ValidationError.from_exception_data("ValidationError", [error_details])
        
        config = MockConfig()
        result = validator.validate(config)
        assert not result.is_valid
        assert len(result.errors) > 0
    
    def test_validate_with_custom_validators(self):
        """Test validation with custom validators."""
        validator = ConfigValidator()
        
        def port_validator(value, config):
            if value < 1 or value > 65535:
                return ValidationIssue(
                    field="port",
                    message="Invalid port",
                    severity=ValidationSeverity.ERROR,
                    value=value
                )
            return None
        
        validator.add_field_validator("server.port", port_validator)
        
        # Create a mock config with invalid port
        class MockConfig:
            def model_dump(self):
                return {"server": {"port": 99999}}
            def model_validate(self, data):
                return data
        
        config = MockConfig()
        result = validator.validate(config)
        assert not result.is_valid
        assert len(result.errors) > 0
    
    def test_validate_with_global_validators(self):
        """Test validation with global validators."""
        validator = ConfigValidator()
        
        def global_validator(config):
            if config.debug and config.environment == Environment.PRODUCTION:
                return ValidationIssue(
                    field="debug",
                    message="Debug should not be enabled in production",
                    severity=ValidationSeverity.ERROR
                )
            return None
        
        validator.add_global_validator(global_validator)
        
        # Create a mock config with production environment and debug enabled
        class MockConfig:
            def model_dump(self):
                return {"debug": True, "environment": "production"}
            def model_validate(self, data):
                return data
            @property
            def debug(self):
                return True
            @property
            def environment(self):
                return Environment.PRODUCTION
        
        config = MockConfig()
        result = validator.validate(config)
        assert not result.is_valid
        assert len(result.errors) > 0


class TestBuiltInValidators:
    """Test built-in validator functions."""
    
    def test_validate_port_valid(self):
        """Test port validation with valid values."""
        assert validate_port(8080, None) is None
        assert validate_port(1, None) is None
        assert validate_port(65535, None) is None
    
    def test_validate_port_invalid(self):
        """Test port validation with invalid values."""
        result = validate_port(0, None)
        assert result is not None
        assert result.severity == ValidationSeverity.ERROR
        assert "Port must be an integer between 1 and 65535" in result.message
        
        result = validate_port(99999, None)
        assert result is not None
        assert result.severity == ValidationSeverity.ERROR
    
    def test_validate_host_valid(self):
        """Test host validation with valid values."""
        assert validate_host("0.0.0.0", None) is None
        assert validate_host("127.0.0.1", None) is None
        assert validate_host("localhost", None) is None
        assert validate_host("example.com", None) is None
    
    def test_validate_host_invalid(self):
        """Test host validation with invalid values."""
        result = validate_host(123, None)
        assert result is not None
        assert result.severity == ValidationSeverity.ERROR
        assert "Host must be a string" in result.message
        
        result = validate_host("invalid@host", None)
        assert result is not None
        assert result.severity == ValidationSeverity.WARNING
    
    def test_validate_file_path_exists(self):
        """Test file path validation with existing file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            temp_file = Path(f.name)
        
        try:
            assert validate_file_path(str(temp_file), None) is None
        finally:
            temp_file.unlink()
    
    def test_validate_file_path_not_exists(self):
        """Test file path validation with non-existing file."""
        result = validate_file_path("/nonexistent/file.txt", None)
        assert result is not None
        assert result.severity == ValidationSeverity.ERROR
        assert "File does not exist" in result.message
    
    def test_validate_file_path_none(self):
        """Test file path validation with None value."""
        assert validate_file_path(None, None) is None
    
    def test_validate_directory_path_exists(self):
        """Test directory path validation with existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            assert validate_directory_path(temp_dir, None) is None
    
    def test_validate_directory_path_not_exists(self):
        """Test directory path validation with non-existing directory."""
        result = validate_directory_path("/nonexistent/dir", None)
        assert result is not None
        assert result.severity == ValidationSeverity.WARNING
        assert "Directory does not exist" in result.message
    
    def test_validate_url_valid(self):
        """Test URL validation with valid URLs."""
        assert validate_url("http://example.com", None) is None
        assert validate_url("https://example.com:8080", None) is None
        assert validate_url("http://localhost:3000", None) is None
        assert validate_url("https://192.168.1.1:9000", None) is None
    
    def test_validate_url_invalid(self):
        """Test URL validation with invalid URLs."""
        result = validate_url("not-a-url", None)
        assert result is not None
        assert result.severity == ValidationSeverity.ERROR
        assert "Invalid URL format" in result.message
        
        result = validate_url(123, None)
        assert result is not None
        assert result.severity == ValidationSeverity.ERROR
        assert "URL must be a string" in result.message
    
    def test_validate_url_none(self):
        """Test URL validation with None value."""
        assert validate_url(None, None) is None


class TestEnvironmentValidation:
    """Test environment-specific validation."""
    
    def test_validate_environment_consistency_production_debug(self):
        """Test production environment validation with debug enabled."""
        # Create a mock config with production environment and debug enabled
        class MockConfig:
            @property
            def environment(self):
                return Environment.PRODUCTION
            @property
            def debug(self):
                return True
            @property
            def server(self):
                return type('Server', (), {'reload': False, 'workers': 1})()
            @property
            def security(self):
                return type('Security', (), {'enabled': True})()
        
        config = MockConfig()
        issues = validate_environment_consistency(config)
        assert len(issues) > 0
        assert any(issue.field == "debug" for issue in issues)
        assert any(issue.severity == ValidationSeverity.ERROR for issue in issues)
    
    def test_validate_environment_consistency_production_reload(self):
        """Test production environment validation with reload enabled."""
        # Create a mock config with production environment and reload enabled
        class MockConfig:
            @property
            def environment(self):
                return Environment.PRODUCTION
            @property
            def debug(self):
                return False
            @property
            def server(self):
                return type('Server', (), {'reload': True, 'workers': 1})()
            @property
            def security(self):
                return type('Security', (), {'enabled': True})()
        
        config = MockConfig()
        issues = validate_environment_consistency(config)
        assert len(issues) > 0
        assert any(issue.field == "server.reload" for issue in issues)
        assert any(issue.severity == ValidationSeverity.ERROR for issue in issues)
    
    def test_validate_environment_consistency_production_security_disabled(self):
        """Test production environment validation with security disabled."""
        config = AppConfig()
        config.environment = Environment.PRODUCTION
        config.security.enabled = False
        
        issues = validate_environment_consistency(config)
        assert len(issues) > 0
        assert any(issue.field == "security.enabled" for issue in issues)
        assert any(issue.severity == ValidationSeverity.WARNING for issue in issues)
    
    def test_validate_environment_consistency_production_single_worker(self):
        """Test production environment validation with single worker."""
        config = AppConfig()
        config.environment = Environment.PRODUCTION
        config.server.workers = 1
        
        issues = validate_environment_consistency(config)
        assert len(issues) > 0
        assert any(issue.field == "server.workers" for issue in issues)
        assert any(issue.severity == ValidationSeverity.INFO for issue in issues)
    
    def test_validate_environment_consistency_development(self):
        """Test development environment validation (should pass)."""
        # Create a mock config with development environment
        class MockConfig:
            @property
            def environment(self):
                return Environment.DEVELOPMENT
            @property
            def debug(self):
                return True
            @property
            def server(self):
                return type('Server', (), {'reload': True, 'workers': 1})()
            @property
            def security(self):
                return type('Security', (), {'enabled': False})()
        
        config = MockConfig()
        issues = validate_environment_consistency(config)
        assert len(issues) == 0


class TestDatabaseValidation:
    """Test database configuration validation."""
    
    def test_validate_database_config_postgresql_missing_host(self):
        """Test PostgreSQL validation with missing host."""
        # Create a mock config with PostgreSQL backend but no host
        class MockConfig:
            @property
            def database(self):
                return type('Database', (), {
                    'backend': DatabaseBackend.POSTGRESQL,
                    'host': None,
                    'url': None,
                    'username': 'test'
                })()
        
        config = MockConfig()
        issues = validate_database_config(config)
        assert len(issues) > 0
        assert any(issue.field == "database.host" for issue in issues)
        assert any(issue.severity == ValidationSeverity.ERROR for issue in issues)
    
    def test_validate_database_config_postgresql_missing_username(self):
        """Test PostgreSQL validation with missing username."""
        # Create a mock config with PostgreSQL backend but no username
        class MockConfig:
            @property
            def database(self):
                return type('Database', (), {
                    'backend': DatabaseBackend.POSTGRESQL,
                    'host': 'localhost',
                    'url': None,
                    'username': None
                })()
        
        config = MockConfig()
        issues = validate_database_config(config)
        assert len(issues) > 0
        assert any(issue.field == "database.username" for issue in issues)
        assert any(issue.severity == ValidationSeverity.ERROR for issue in issues)
    
    def test_validate_database_config_sqlite(self):
        """Test SQLite validation (should pass)."""
        # Create a mock config with SQLite backend
        class MockConfig:
            @property
            def database(self):
                return type('Database', (), {
                    'backend': DatabaseBackend.SQLITE,
                    'host': None,
                    'url': None,
                    'username': None
                })()
        
        config = MockConfig()
        issues = validate_database_config(config)
        assert len(issues) == 0


class TestCacheValidation:
    """Test cache configuration validation."""
    
    def test_validate_cache_config_redis_missing_host(self):
        """Test Redis validation with missing host."""
        # Create a mock config with Redis backend but no host
        class MockConfig:
            @property
            def cache(self):
                return type('Cache', (), {
                    'backend': CacheBackend.REDIS,
                    'host': None
                })()
        
        config = MockConfig()
        issues = validate_cache_config(config)
        assert len(issues) > 0
        assert any(issue.field == "cache.host" for issue in issues)
        assert any(issue.severity == ValidationSeverity.ERROR for issue in issues)
    
    def test_validate_cache_config_memory(self):
        """Test memory cache validation (should pass)."""
        # Create a mock config with memory cache backend
        class MockConfig:
            @property
            def cache(self):
                return type('Cache', (), {
                    'backend': CacheBackend.MEMORY,
                    'host': None
                })()
        
        config = MockConfig()
        issues = validate_cache_config(config)
        assert len(issues) == 0


class TestModelValidation:
    """Test model configuration validation."""
    
    def test_validate_model_config_invalid_path(self):
        """Test model validation with invalid path."""
        # Create a mock config with invalid model path
        class MockConfig:
            @property
            def model(self):
                return type('Model', (), {
                    'model_path': Path("/nonexistent/model/path"),
                    'model_file': None,
                    'batch_size': 1,
                    'max_batch_size': 32
                })()
        
        config = MockConfig()
        issues = validate_model_config(config)
        assert len(issues) > 0
        assert any(issue.field == "model.model_path" for issue in issues)
        assert any(issue.severity == ValidationSeverity.ERROR for issue in issues)
    
    def test_validate_model_config_invalid_file(self):
        """Test model validation with invalid file."""
        # Create a mock config with invalid model file
        class MockConfig:
            @property
            def model(self):
                return type('Model', (), {
                    'model_path': Path("/tmp"),  # Valid path
                    'model_file': Path("/nonexistent/model.pkl"),
                    'batch_size': 1,
                    'max_batch_size': 32
                })()
        
        config = MockConfig()
        issues = validate_model_config(config)
        assert len(issues) > 0
        assert any(issue.field == "model.model_file" for issue in issues)
        assert any(issue.severity == ValidationSeverity.ERROR for issue in issues)
    
    def test_validate_model_config_invalid_batch_size(self):
        """Test model validation with invalid batch size."""
        # Create a mock config with invalid batch size configuration
        class MockConfig:
            @property
            def model(self):
                return type('Model', (), {
                    'model_path': Path("/tmp"),  # Valid path
                    'model_file': None,
                    'batch_size': 10,
                    'max_batch_size': 5  # Invalid: max < batch
                })()
        
        config = MockConfig()
        issues = validate_model_config(config)
        assert len(issues) > 0
        assert any(issue.field == "model.max_batch_size" for issue in issues)
        assert any(issue.severity == ValidationSeverity.ERROR for issue in issues)


class TestDefaultValidator:
    """Test default validator creation."""
    
    def test_create_default_validator(self):
        """Test creating default validator."""
        validator = create_default_validator()
        assert isinstance(validator, ConfigValidator)
        assert len(validator._validators) > 0
        assert len(validator._custom_rules) > 0


class TestValidateConfig:
    """Test validate_config function."""
    
    def test_validate_config_with_default_validator(self):
        """Test validation with default validator."""
        config = AppConfig()
        result = validate_config(config)
        assert isinstance(result, ValidationResult)
    
    def test_validate_config_with_custom_validator(self):
        """Test validation with custom validator."""
        validator = ConfigValidator()
        
        def custom_validator(config):
            return [ValidationIssue("test", "Custom error", ValidationSeverity.ERROR)]
        
        validator.add_global_validator(custom_validator)
        
        config = AppConfig()
        result = validate_config(config, validator)
        assert not result.is_valid
        assert len(result.errors) > 0


class TestFormatValidationResult:
    """Test validation result formatting."""
    
    def test_format_validation_result(self):
        """Test formatting validation result."""
        issues = [
            ValidationIssue("field1", "Error message", ValidationSeverity.ERROR),
            ValidationIssue("field2", "Warning message", ValidationSeverity.WARNING),
            ValidationIssue("field3", "Info message", ValidationSeverity.INFO)
        ]
        
        result = ValidationResult(
            is_valid=False,
            issues=issues,
            warnings=[],
            errors=[],
            summary={}
        )
        
        formatted = format_validation_result(result)
        assert "Configuration Validation Result" in formatted
        assert "Valid: False" in formatted
        assert "ERRORS:" in formatted
        assert "WARNINGS:" in formatted
        assert "INFOS:" in formatted
        assert "field1" in formatted
        assert "field2" in formatted
        assert "field3" in formatted


class TestValidationError:
    """Test ValidationError exception."""
    
    def test_validation_error_creation(self):
        """Test creating validation error."""
        result = ValidationResult(True, [], [], [], {})
        error = ValidationError("Test error", result)
        
        assert str(error) == "Test error"
        assert error.validation_result == result
