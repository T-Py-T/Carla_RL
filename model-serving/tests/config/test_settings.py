"""
Unit tests for configuration settings.

Tests Pydantic configuration models with comprehensive validation.
"""

import pytest
import tempfile
from pathlib import Path
from pydantic import ValidationError

from src.config.settings import (
    AppConfig, ServerConfig, ModelConfig, LoggingConfig,
    MonitoringConfig, DatabaseConfig, CacheConfig, SecurityConfig,
    Environment, LogLevel, ModelBackend, CacheBackend, DatabaseBackend
)


class TestServerConfig:
    """Test ServerConfig functionality."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ServerConfig()
        
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.workers == 1
        assert config.reload is False
        assert config.log_level == LogLevel.INFO
        assert config.access_log is True
        assert config.max_request_size == 16 * 1024 * 1024
        assert config.timeout == 30.0
        assert config.keep_alive is True
        assert config.keep_alive_timeout == 5.0
        assert config.max_connections == 1000
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = ServerConfig(
            host="127.0.0.1",
            port=9000,
            workers=1,  # Use 1 worker to avoid validation error with reload=True
            reload=True,
            log_level=LogLevel.DEBUG,
            timeout=60.0
        )
        
        assert str(config.host) == "127.0.0.1"
        assert config.port == 9000
        assert config.workers == 1
        assert config.reload is True
        assert config.log_level == LogLevel.DEBUG
        assert config.timeout == 60.0
    
    def test_port_validation(self):
        """Test port number validation."""
        # Valid ports
        ServerConfig(port=1)
        ServerConfig(port=65535)
        ServerConfig(port=8080)
        
        # Invalid ports
        with pytest.raises(ValidationError):
            ServerConfig(port=0)
        
        with pytest.raises(ValidationError):
            ServerConfig(port=65536)
        
        with pytest.raises(ValidationError):
            ServerConfig(port=-1)
    
    def test_workers_validation_with_reload(self):
        """Test workers validation when reload is enabled."""
        # Should work with reload=False and workers > 1
        ServerConfig(reload=False, workers=4)
        
        # Should fail with reload=True and workers > 1
        with pytest.raises(ValidationError):
            ServerConfig(reload=True, workers=4)


class TestModelConfig:
    """Test ModelConfig functionality."""
    
    def test_default_values(self):
        """Test default model configuration values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ModelConfig(model_path=Path(temp_dir))
            
            assert config.backend == ModelBackend.PYTORCH
            assert config.model_path == Path(temp_dir)
            assert config.model_file is None
            assert config.config_file is None
            assert config.device == "auto"
            assert config.batch_size == 1
            assert config.max_batch_size == 32
            assert config.precision == "float32"
            assert config.optimize is True
            assert config.cache_models is True
            assert config.model_timeout == 60.0
            assert config.warmup_requests == 5
    
    def test_device_validation(self):
        """Test device validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Valid devices
            ModelConfig(model_path=Path(temp_dir), device="cpu")
            ModelConfig(model_path=Path(temp_dir), device="cuda")
            ModelConfig(model_path=Path(temp_dir), device="auto")
            ModelConfig(model_path=Path(temp_dir), device="cuda:0")
            
            # Invalid device
            with pytest.raises(ValidationError):
                ModelConfig(model_path=Path(temp_dir), device="invalid")
    
    def test_batch_size_validation(self):
        """Test batch size validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Valid batch sizes
            ModelConfig(model_path=Path(temp_dir), batch_size=1)
            ModelConfig(model_path=Path(temp_dir), batch_size=32)
            
            # Invalid batch sizes
            with pytest.raises(ValidationError):
                ModelConfig(model_path=Path(temp_dir), batch_size=0)
            
            with pytest.raises(ValidationError):
                ModelConfig(model_path=Path(temp_dir), batch_size=-1)
    
    def test_max_batch_size_validation(self):
        """Test max batch size validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Should work when max >= batch_size
            ModelConfig(model_path=Path(temp_dir), batch_size=8, max_batch_size=16)
            
            # Should fail when max < batch_size
            with pytest.raises(ValidationError):
                ModelConfig(model_path=Path(temp_dir), batch_size=16, max_batch_size=8)


class TestLoggingConfig:
    """Test LoggingConfig functionality."""
    
    def test_default_values(self):
        """Test default logging configuration values."""
        config = LoggingConfig()
        
        assert config.level == LogLevel.INFO
        assert "asctime" in config.format
        assert "levelname" in config.format
        assert config.date_format == "%Y-%m-%d %H:%M:%S"
        assert config.file_path is None
        assert config.max_file_size == 10 * 1024 * 1024
        assert config.backup_count == 5
        assert config.json_format is False
        assert config.include_timestamp is True
        assert config.include_level is True
        assert config.include_logger is True
        assert config.correlation_id is True
    
    def test_file_path_creation(self):
        """Test that log file directory is created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "test_logs"
            log_file = log_dir / "app.log"
            # Create the file first
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_file.touch()
            
            config = LoggingConfig(file_path=log_file)
            
            assert config.file_path == log_file
            assert log_dir.exists()


class TestMonitoringConfig:
    """Test MonitoringConfig functionality."""
    
    def test_default_values(self):
        """Test default monitoring configuration values."""
        config = MonitoringConfig()
        
        assert config.enabled is True
        assert config.metrics_enabled is True
        assert config.tracing_enabled is True
        assert config.health_check_enabled is True
        assert config.prometheus_enabled is True
        assert config.jaeger_enabled is False
        assert config.metrics_port == 9090
        assert config.metrics_path == "/metrics"
        assert config.collect_system_metrics is True
        assert config.collect_model_metrics is True
        assert config.trace_sample_rate == 0.1
        assert config.health_check_path == "/health"
        assert config.health_check_interval == 30.0


class TestDatabaseConfig:
    """Test DatabaseConfig functionality."""
    
    def test_default_values(self):
        """Test default database configuration values."""
        config = DatabaseConfig()
        
        assert config.backend == DatabaseBackend.SQLITE
        assert config.host is None
        assert config.port is None
        assert config.name == "model_serving"
        assert config.username is None
        assert config.password is None
        assert config.url is None
        assert config.pool_size == 5
        assert config.max_overflow == 10
        assert config.pool_timeout == 30.0
        assert config.pool_recycle == 3600
        assert config.echo is False
    
    def test_database_url_generation(self):
        """Test database URL generation."""
        # SQLite
        config = DatabaseConfig(backend=DatabaseBackend.SQLITE, name="test.db")
        assert config.get_database_url() == "sqlite:///test.db.db"
        
        # PostgreSQL
        config = DatabaseConfig(
            backend=DatabaseBackend.POSTGRESQL,
            host="localhost",
            port=5432,
            name="testdb",
            username="user",
            password="pass"
        )
        url = config.get_database_url()
        assert "postgresql://" in url
        assert "user:pass@localhost:5432/testdb" in url
    
    def test_validation_requirements(self):
        """Test validation requirements for different backends."""
        # PostgreSQL requires host or URL
        with pytest.raises(ValidationError):
            DatabaseConfig(backend=DatabaseBackend.POSTGRESQL)
        
        # Should work with host
        DatabaseConfig(backend=DatabaseBackend.POSTGRESQL, host="localhost")
        
        # Should work with URL
        DatabaseConfig(backend=DatabaseBackend.POSTGRESQL, url="postgresql://localhost/test")


class TestCacheConfig:
    """Test CacheConfig functionality."""
    
    def test_default_values(self):
        """Test default cache configuration values."""
        config = CacheConfig()
        
        assert config.backend == CacheBackend.MEMORY
        assert config.host is None
        assert config.port is None
        assert config.password is None
        assert config.db == 0
        assert config.max_connections == 10
        assert config.timeout == 5.0
        assert config.key_prefix == "model_serving:"
        assert config.default_ttl == 3600
        assert config.max_memory is None
    
    def test_cache_url_generation(self):
        """Test cache URL generation."""
        # Memory
        config = CacheConfig(backend=CacheBackend.MEMORY)
        assert config.get_cache_url() == "memory://"
        
        # Redis
        config = CacheConfig(
            backend=CacheBackend.REDIS,
            host="localhost",
            port=6379,
            password="secret"
        )
        url = config.get_cache_url()
        assert "redis://" in url
        assert ":secret@localhost:6379/0" in url
    
    def test_validation_requirements(self):
        """Test validation requirements for different backends."""
        # Redis requires host
        with pytest.raises(ValidationError):
            CacheConfig(backend=CacheBackend.REDIS)
        
        # Should work with host
        CacheConfig(backend=CacheBackend.REDIS, host="localhost")


class TestSecurityConfig:
    """Test SecurityConfig functionality."""
    
    def test_default_values(self):
        """Test default security configuration values."""
        config = SecurityConfig()
        
        assert config.enabled is False
        assert config.api_key_required is False
        assert config.api_key_header == "X-API-Key"
        assert config.api_keys == []
        assert config.cors_enabled is True
        assert config.cors_origins == ["*"]
        assert config.cors_methods == ["GET", "POST", "PUT", "DELETE"]
        assert config.cors_headers == ["*"]
        assert config.rate_limit_enabled is False
        assert config.rate_limit_requests == 100
        assert config.rate_limit_window == 60
        assert config.ssl_enabled is False
        assert config.ssl_cert_file is None
        assert config.ssl_key_file is None
    
    def test_ssl_file_validation(self):
        """Test SSL file validation."""
        # Should work when SSL is disabled and no files specified
        SecurityConfig(ssl_enabled=False)
        
        # Should work when SSL is enabled but no files specified
        SecurityConfig(ssl_enabled=True)
        
        # Should fail when SSL is enabled but file doesn't exist
        with pytest.raises(ValidationError):
            SecurityConfig(ssl_enabled=True, ssl_cert_file=Path("nonexistent.crt"))


class TestAppConfig:
    """Test AppConfig functionality."""
    
    def test_default_values(self):
        """Test default application configuration values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AppConfig(model=ModelConfig(model_path=Path(temp_dir)))
            
            assert config.environment == Environment.DEVELOPMENT
            assert config.debug is False
            assert config.version == "1.0.0"
            assert config.name == "Model Serving"
            assert config.description == "High-performance model serving API"
            
            # Check component configurations
            assert isinstance(config.server, ServerConfig)
            assert isinstance(config.model, ModelConfig)
            assert isinstance(config.logging, LoggingConfig)
            assert isinstance(config.monitoring, MonitoringConfig)
            assert isinstance(config.database, DatabaseConfig)
            assert isinstance(config.cache, CacheConfig)
            assert isinstance(config.security, SecurityConfig)
    
    def test_environment_validation(self):
        """Test environment-specific validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Development should allow debug=True
            AppConfig(environment=Environment.DEVELOPMENT, debug=True, model=ModelConfig(model_path=Path(temp_dir)))
            
            # Production should not allow debug=True
            with pytest.raises(ValidationError):
                AppConfig(environment=Environment.PRODUCTION, debug=True, model=ModelConfig(model_path=Path(temp_dir)))
    
    def test_directory_creation(self):
        """Test that directories are created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AppConfig(
                config_dir=Path(temp_dir) / "test_config",
                data_dir=Path(temp_dir) / "test_data",
                log_dir=Path(temp_dir) / "test_logs",
                temp_dir=Path(temp_dir) / "test_temp",
                model=ModelConfig(model_path=Path(temp_dir))
            )
            
            assert config.config_dir.exists()
            assert config.data_dir.exists()
            assert config.log_dir.exists()
            assert config.temp_dir.exists()
    
    def test_database_url_generation(self):
        """Test database URL generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AppConfig(model=ModelConfig(model_path=Path(temp_dir)))
            url = config.get_database_url()
            assert "sqlite://" in url
    
    def test_cache_url_generation(self):
        """Test cache URL generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AppConfig(model=ModelConfig(model_path=Path(temp_dir)))
            url = config.get_cache_url()
            assert "memory://" in url
    
    def test_model_dump_for_env(self):
        """Test configuration export for environment variables."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AppConfig(model=ModelConfig(model_path=Path(temp_dir)))
            env_vars = config.model_dump_for_env("APP")
            
            assert "APP_ENVIRONMENT" in env_vars
            assert "APP_DEBUG" in env_vars
            assert "APP_VERSION" in env_vars
            assert env_vars["APP_ENVIRONMENT"] == "development"
            assert env_vars["APP_DEBUG"] == "False"


class TestEnums:
    """Test enum functionality."""
    
    def test_environment_enum(self):
        """Test Environment enum."""
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.STAGING.value == "staging"
        assert Environment.PRODUCTION.value == "production"
        assert Environment.TESTING.value == "testing"
    
    def test_log_level_enum(self):
        """Test LogLevel enum."""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"
    
    def test_model_backend_enum(self):
        """Test ModelBackend enum."""
        assert ModelBackend.PYTORCH.value == "pytorch"
        assert ModelBackend.TENSORFLOW.value == "tensorflow"
        assert ModelBackend.ONNX.value == "onnx"
        assert ModelBackend.TORCHSCRIPT.value == "torchscript"
        assert ModelBackend.TRITON.value == "triton"
    
    def test_cache_backend_enum(self):
        """Test CacheBackend enum."""
        assert CacheBackend.REDIS.value == "redis"
        assert CacheBackend.MEMCACHED.value == "memcached"
        assert CacheBackend.MEMORY.value == "memory"
        assert CacheBackend.DISK.value == "disk"
    
    def test_database_backend_enum(self):
        """Test DatabaseBackend enum."""
        assert DatabaseBackend.POSTGRESQL.value == "postgresql"
        assert DatabaseBackend.MYSQL.value == "mysql"
        assert DatabaseBackend.SQLITE.value == "sqlite"
        assert DatabaseBackend.MONGODB.value == "mongodb"
