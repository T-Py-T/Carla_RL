"""
Pydantic configuration models with comprehensive validation.

Provides type-safe configuration models for all aspects of the model serving system.
"""

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
    HttpUrl,
    IPvAnyAddress,
    DirectoryPath,
    FilePath,
    SecretStr,
)
from pydantic.networks import AnyHttpUrl, HttpUrl
from pydantic import PositiveInt, NonNegativeInt, PositiveFloat


class Environment(str, Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ModelBackend(str, Enum):
    """Model backend types."""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    TRITON = "triton"


class CacheBackend(str, Enum):
    """Cache backend types."""
    REDIS = "redis"
    MEMCACHED = "memcached"
    MEMORY = "memory"
    DISK = "disk"


class DatabaseBackend(str, Enum):
    """Database backend types."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    MONGODB = "mongodb"


class BaseConfig(BaseModel):
    """Base configuration class with common settings."""
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        validate_assignment=True,
        use_enum_values=True,
        extra="forbid"
    )
        
    def model_dump_for_env(self, prefix: str = "") -> Dict[str, str]:
        """Convert config to environment variables format."""
        result = {}
        for key, value in self.model_dump().items():
            env_key = f"{prefix}_{key}".upper() if prefix else key.upper()
            if isinstance(value, (dict, list)):
                result[env_key] = str(value)
            elif value is not None:
                # Convert enum values to their string representation
                if hasattr(value, 'value'):
                    result[env_key] = str(value.value)
                else:
                    result[env_key] = str(value)
        return result


class ServerConfig(BaseConfig):
    """Server configuration."""
    
    host: IPvAnyAddress = Field(default="0.0.0.0", description="Server host address")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    workers: PositiveInt = Field(default=1, description="Number of worker processes")
    reload: bool = Field(default=False, description="Enable auto-reload in development")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    access_log: bool = Field(default=True, description="Enable access logging")
    max_request_size: PositiveInt = Field(
        default=16 * 1024 * 1024,  # 16MB
        description="Maximum request size in bytes"
    )
    timeout: PositiveFloat = Field(default=30.0, description="Request timeout in seconds")
    keep_alive: bool = Field(default=True, description="Enable keep-alive connections")
    keep_alive_timeout: PositiveFloat = Field(
        default=5.0, 
        description="Keep-alive timeout in seconds"
    )
    max_connections: PositiveInt = Field(
        default=1000, 
        description="Maximum concurrent connections"
    )
    
    @model_validator(mode='after')
    def validate_workers_with_reload(self):
        """Validate worker count based on reload setting."""
        if self.reload and self.workers > 1:
            raise ValueError("Cannot use multiple workers with reload enabled")
        return self


class ModelConfig(BaseConfig):
    """Model configuration."""
    
    backend: ModelBackend = Field(default=ModelBackend.PYTORCH, description="Model backend")
    model_path: Optional[DirectoryPath] = Field(default=None, description="Path to model directory")
    model_file: Optional[FilePath] = Field(default=None, description="Specific model file")
    config_file: Optional[FilePath] = Field(default=None, description="Model configuration file")
    device: str = Field(default="auto", description="Device to run model on (cpu, cuda, auto)")
    batch_size: PositiveInt = Field(default=1, description="Default batch size")
    max_batch_size: PositiveInt = Field(default=32, description="Maximum batch size")
    precision: Literal["float32", "float16", "int8"] = Field(
        default="float32", 
        description="Model precision"
    )
    optimize: bool = Field(default=True, description="Enable model optimization")
    cache_models: bool = Field(default=True, description="Cache loaded models in memory")
    model_timeout: PositiveFloat = Field(
        default=60.0, 
        description="Model loading timeout in seconds"
    )
    warmup_requests: NonNegativeInt = Field(
        default=5, 
        description="Number of warmup requests"
    )
    
    @field_validator("device")
    @classmethod
    def validate_device(cls, v):
        """Validate device specification."""
        valid_devices = ["cpu", "cuda", "auto", "cuda:0", "cuda:1", "cuda:2", "cuda:3"]
        if v not in valid_devices and not v.startswith("cuda:"):
            raise ValueError(f"Invalid device: {v}. Must be one of {valid_devices}")
        return v
    
    @field_validator("max_batch_size")
    @classmethod
    def validate_max_batch_size(cls, v, info):
        """Validate max batch size is >= batch size."""
        if hasattr(info, 'data') and 'batch_size' in info.data and v < info.data["batch_size"]:
            raise ValueError("max_batch_size must be >= batch_size")
        return v


class LoggingConfig(BaseConfig):
    """Logging configuration."""
    
    level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    date_format: str = Field(
        default="%Y-%m-%d %H:%M:%S",
        description="Date format in logs"
    )
    file_path: Optional[FilePath] = Field(default=None, description="Log file path")
    max_file_size: PositiveInt = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="Maximum log file size in bytes"
    )
    backup_count: NonNegativeInt = Field(
        default=5, 
        description="Number of backup log files"
    )
    json_format: bool = Field(default=False, description="Use JSON log format")
    include_timestamp: bool = Field(default=True, description="Include timestamp in logs")
    include_level: bool = Field(default=True, description="Include level in logs")
    include_logger: bool = Field(default=True, description="Include logger name in logs")
    correlation_id: bool = Field(default=True, description="Include correlation ID in logs")
    
    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v):
        """Ensure log file directory exists."""
        if v and not v.parent.exists():
            v.parent.mkdir(parents=True, exist_ok=True)
        return v


class MonitoringConfig(BaseConfig):
    """Monitoring and observability configuration."""
    
    enabled: bool = Field(default=True, description="Enable monitoring")
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    tracing_enabled: bool = Field(default=True, description="Enable distributed tracing")
    health_check_enabled: bool = Field(default=True, description="Enable health checks")
    prometheus_enabled: bool = Field(default=True, description="Enable Prometheus metrics")
    jaeger_enabled: bool = Field(default=False, description="Enable Jaeger tracing")
    
    # Metrics configuration
    metrics_port: int = Field(
        default=9090, 
        ge=1, 
        le=65535, 
        description="Prometheus metrics port"
    )
    metrics_path: str = Field(default="/metrics", description="Metrics endpoint path")
    collect_system_metrics: bool = Field(
        default=True, 
        description="Collect system-level metrics"
    )
    collect_model_metrics: bool = Field(
        default=True, 
        description="Collect model-specific metrics"
    )
    
    # Tracing configuration
    jaeger_endpoint: Optional[HttpUrl] = Field(
        default=None, 
        description="Jaeger collector endpoint"
    )
    trace_sample_rate: float = Field(
        default=0.1, 
        ge=0.0, 
        le=1.0, 
        description="Trace sampling rate"
    )
    
    # Health check configuration
    health_check_path: str = Field(default="/health", description="Health check endpoint")
    health_check_interval: PositiveFloat = Field(
        default=30.0, 
        description="Health check interval in seconds"
    )
    readiness_probe_path: str = Field(
        default="/ready", 
        description="Readiness probe endpoint"
    )
    liveness_probe_path: str = Field(
        default="/live", 
        description="Liveness probe endpoint"
    )


class DatabaseConfig(BaseConfig):
    """Database configuration."""
    
    backend: DatabaseBackend = Field(
        default=DatabaseBackend.SQLITE, 
        description="Database backend"
    )
    host: Optional[str] = Field(default=None, description="Database host")
    port: Optional[int] = Field(default=None, ge=1, le=65535, description="Database port")
    name: str = Field(default="model_serving", description="Database name")
    username: Optional[str] = Field(default=None, description="Database username")
    password: Optional[SecretStr] = Field(default=None, description="Database password")
    url: Optional[str] = Field(default=None, description="Database connection URL")
    pool_size: PositiveInt = Field(default=5, description="Connection pool size")
    max_overflow: NonNegativeInt = Field(
        default=10, 
        description="Maximum pool overflow"
    )
    pool_timeout: PositiveFloat = Field(
        default=30.0, 
        description="Pool timeout in seconds"
    )
    pool_recycle: PositiveInt = Field(
        default=3600, 
        description="Pool recycle time in seconds"
    )
    echo: bool = Field(default=False, description="Echo SQL queries")
    
    @model_validator(mode='after')
    def validate_database_config(self):
        """Validate database configuration based on backend."""
        if self.backend in [DatabaseBackend.POSTGRESQL, DatabaseBackend.MYSQL]:
            if not self.url and not self.host:
                raise ValueError(f"Host or URL required for {self.backend} backend")
        elif self.backend == DatabaseBackend.SQLITE:
            # SQLite doesn't need host/port
            pass
            
        return self
    
    def get_database_url(self) -> str:
        """Get database connection URL."""
        if self.url:
            return str(self.url)
        
        if self.backend == DatabaseBackend.SQLITE:
            return f"sqlite:///{self.name}.db"
        
        # Build URL for other backends
        host = self.host or "localhost"
        port = self.port or self._get_default_port()
        username = self.username or ""
        password = self.password.get_secret_value() if self.password else ""
        
        if username and password:
            auth = f"{username}:{password}@"
        elif username:
            auth = f"{username}@"
        else:
            auth = ""
        
        return f"{self.backend}://{auth}{host}:{port}/{self.name}"
    
    def _get_default_port(self) -> int:
        """Get default port for database backend."""
        defaults = {
            DatabaseBackend.POSTGRESQL: 5432,
            DatabaseBackend.MYSQL: 3306,
            DatabaseBackend.MONGODB: 27017,
        }
        return defaults.get(self.backend, 5432)


class CacheConfig(BaseConfig):
    """Cache configuration."""
    
    backend: CacheBackend = Field(default=CacheBackend.MEMORY, description="Cache backend")
    host: Optional[str] = Field(default=None, description="Cache host")
    port: Optional[int] = Field(default=None, ge=1, le=65535, description="Cache port")
    password: Optional[SecretStr] = Field(default=None, description="Cache password")
    db: NonNegativeInt = Field(default=0, description="Cache database number")
    max_connections: PositiveInt = Field(
        default=10, 
        description="Maximum cache connections"
    )
    timeout: PositiveFloat = Field(default=5.0, description="Cache timeout in seconds")
    key_prefix: str = Field(default="model_serving:", description="Cache key prefix")
    default_ttl: PositiveInt = Field(
        default=3600, 
        description="Default TTL in seconds"
    )
    max_memory: Optional[str] = Field(
        default=None, 
        description="Maximum memory usage (e.g., '100MB')"
    )
    
    @model_validator(mode='after')
    def validate_cache_backend(self):
        """Validate cache backend configuration."""
        if self.backend in [CacheBackend.REDIS, CacheBackend.MEMCACHED]:
            if not self.host:
                raise ValueError(f"Host required for {self.backend} backend")
        return self
    
    def get_cache_url(self) -> str:
        """Get cache connection URL."""
        if self.backend == CacheBackend.MEMORY:
            return "memory://"
        
        host = self.host or "localhost"
        port = self.port or self._get_default_cache_port()
        password = self.password.get_secret_value() if self.password else ""
        
        if password:
            return f"{self.backend}://:{password}@{host}:{port}/{self.db}"
        else:
            return f"{self.backend}://{host}:{port}/{self.db}"
    
    def _get_default_cache_port(self) -> int:
        """Get default port for cache backend."""
        defaults = {
            CacheBackend.REDIS: 6379,
            CacheBackend.MEMCACHED: 11211,
        }
        return defaults.get(self.backend, 6379)


class SecurityConfig(BaseConfig):
    """Security configuration."""
    
    enabled: bool = Field(default=False, description="Enable security features")
    api_key_required: bool = Field(default=False, description="Require API key")
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    api_keys: List[SecretStr] = Field(default_factory=list, description="Valid API keys")
    cors_enabled: bool = Field(default=True, description="Enable CORS")
    cors_origins: List[str] = Field(
        default=["*"], 
        description="Allowed CORS origins"
    )
    cors_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE"],
        description="Allowed CORS methods"
    )
    cors_headers: List[str] = Field(
        default=["*"],
        description="Allowed CORS headers"
    )
    rate_limit_enabled: bool = Field(default=False, description="Enable rate limiting")
    rate_limit_requests: PositiveInt = Field(
        default=100, 
        description="Rate limit requests per minute"
    )
    rate_limit_window: PositiveInt = Field(
        default=60, 
        description="Rate limit window in seconds"
    )
    ssl_enabled: bool = Field(default=False, description="Enable SSL/TLS")
    ssl_cert_file: Optional[FilePath] = Field(
        default=None, 
        description="SSL certificate file"
    )
    ssl_key_file: Optional[FilePath] = Field(
        default=None, 
        description="SSL private key file"
    )
    
    @model_validator(mode='after')
    def validate_ssl_files(self):
        """Validate SSL files exist when SSL is enabled."""
        if self.ssl_enabled and self.ssl_cert_file and not self.ssl_cert_file.exists():
            raise ValueError(f"SSL certificate file not found: {self.ssl_cert_file}")
        if self.ssl_enabled and self.ssl_key_file and not self.ssl_key_file.exists():
            raise ValueError(f"SSL key file not found: {self.ssl_key_file}")
        return self


class AppConfig(BaseConfig):
    """Main application configuration."""
    
    # Environment
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment"
    )
    debug: bool = Field(default=False, description="Debug mode")
    version: str = Field(default="1.0.0", description="Application version")
    name: str = Field(default="Model Serving", description="Application name")
    description: str = Field(
        default="High-performance model serving API",
        description="Application description"
    )
    
    # Component configurations
    server: ServerConfig = Field(default_factory=ServerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    
    # File paths
    config_dir: Path = Field(
        default=Path("config"),
        description="Configuration directory"
    )
    data_dir: Path = Field(
        default=Path("data"),
        description="Data directory"
    )
    log_dir: Path = Field(
        default=Path("logs"),
        description="Log directory"
    )
    temp_dir: Path = Field(
        default=Path("temp"),
        description="Temporary directory"
    )
    
    @field_validator("debug")
    @classmethod
    def validate_debug_mode(cls, v, info):
        """Validate debug mode based on environment."""
        if hasattr(info, 'data') and info.data.get("environment") == Environment.PRODUCTION and v:
            raise ValueError("Debug mode cannot be enabled in production")
        return v
    
    @model_validator(mode='after')
    def ensure_directories_exist(self):
        """Ensure directories exist."""
        for dir_field in ["config_dir", "data_dir", "log_dir", "temp_dir"]:
            dir_path = getattr(self, dir_field)
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
        return self
    
    @model_validator(mode='after')
    def validate_configuration(self):
        """Validate overall configuration consistency."""
        # Production-specific validations
        if self.environment == Environment.PRODUCTION:
            if self.server.reload:
                raise ValueError("Auto-reload cannot be enabled in production")
            
            if not self.security.enabled:
                raise ValueError("Security must be enabled in production")
        
        return self
    
    def get_database_url(self) -> str:
        """Get database connection URL."""
        db_config = self.database
        
        if db_config.url:
            return str(db_config.url)
        
        if db_config.backend == DatabaseBackend.SQLITE:
            return f"sqlite:///{db_config.name}.db"
        
        # Build URL for other backends
        host = db_config.host or "localhost"
        port = db_config.port or self._get_default_port(db_config.backend)
        username = db_config.username or ""
        password = db_config.password.get_secret_value() if db_config.password else ""
        
        if username and password:
            auth = f"{username}:{password}@"
        elif username:
            auth = f"{username}@"
        else:
            auth = ""
        
        return f"{db_config.backend.value}://{auth}{host}:{port}/{db_config.name}"
    
    def _get_default_port(self, backend: DatabaseBackend) -> int:
        """Get default port for database backend."""
        defaults = {
            DatabaseBackend.POSTGRESQL: 5432,
            DatabaseBackend.MYSQL: 3306,
            DatabaseBackend.MONGODB: 27017,
        }
        return defaults.get(backend, 5432)
    
    def get_cache_url(self) -> str:
        """Get cache connection URL."""
        cache_config = self.cache
        
        if cache_config.backend == CacheBackend.MEMORY:
            return "memory://"
        
        host = cache_config.host or "localhost"
        port = cache_config.port or self._get_default_cache_port(cache_config.backend)
        password = cache_config.password.get_secret_value() if cache_config.password else ""
        
        if password:
            return f"{cache_config.backend.value}://:{password}@{host}:{port}/{cache_config.db}"
        else:
            return f"{cache_config.backend.value}://{host}:{port}/{cache_config.db}"
    
    def _get_default_cache_port(self, backend: CacheBackend) -> int:
        """Get default port for cache backend."""
        defaults = {
            CacheBackend.REDIS: 6379,
            CacheBackend.MEMCACHED: 11211,
        }
        return defaults.get(backend, 6379)
