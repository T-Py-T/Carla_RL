"""
Pydantic configuration models with comprehensive validation.

Defines all configuration models for the model serving application
with proper validation, type hints, and documentation.
"""

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from pydantic import HttpUrl, IPvAnyAddress, DirectoryPath, FilePath, SecretStr


class Environment(str, Enum):
    """Application environment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


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
    TRITON = "triton"


class DatabaseBackend(str, Enum):
    """Database backend types."""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"


class CacheBackend(str, Enum):
    """Cache backend types."""
    MEMORY = "memory"
    REDIS = "redis"
    MEMCACHED = "memcached"


class BaseConfig(BaseModel):
    """Base configuration class with common functionality."""
    
    model_config = ConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        validate_assignment=True,
        use_enum_values=True,
        extra="forbid"
    )
    
    def model_dump_for_env(self) -> Dict[str, str]:
        """Convert model to environment variable format."""
        result = {}
        for key, value in self.model_dump(exclude_unset=True).items():
            if isinstance(value, Enum):
                result[key.upper()] = value.value
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    result[f"{key.upper()}_{sub_key.upper()}"] = str(sub_value)
            else:
                result[key.upper()] = str(value)
        return result


class ServerConfig(BaseModel):
    """Server configuration."""
    
    model_config = ConfigDict(extra="forbid")
    
    host: IPvAnyAddress = Field(default="0.0.0.0", description="Server host address")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    workers: int = Field(default=1, ge=1, description="Number of worker processes")
    reload: bool = Field(default=False, description="Enable auto-reload")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Log level")
    access_log: bool = Field(default=True, description="Enable access logging")
    max_request_size: int = Field(default=16 * 1024 * 1024, ge=1, description="Max request size in bytes")
    timeout: float = Field(default=30.0, ge=0, description="Request timeout in seconds")
    keep_alive: bool = Field(default=True, description="Enable keep-alive")
    keep_alive_timeout: float = Field(default=5.0, ge=0, description="Keep-alive timeout")
    max_connections: int = Field(default=1000, ge=1, description="Max concurrent connections")
    
    @model_validator(mode='after')
    def validate_workers(self):
        """Validate workers configuration."""
        if self.reload and self.workers > 1:
            raise ValueError("Cannot use multiple workers with reload enabled")
        return self


class ModelConfig(BaseModel):
    """Model configuration."""
    
    model_config = ConfigDict(extra="forbid")
    
    backend: ModelBackend = Field(default=ModelBackend.PYTORCH, description="Model backend")
    model_path: Optional[DirectoryPath] = Field(default=None, description="Path to model directory")
    model_file: Optional[FilePath] = Field(default=None, description="Path to model file")
    config_file: Optional[FilePath] = Field(default=None, description="Path to model config file")
    device: str = Field(default="auto", description="Device to run model on")
    batch_size: int = Field(default=1, ge=1, description="Default batch size")
    max_batch_size: int = Field(default=32, ge=1, description="Maximum batch size")
    precision: str = Field(default="float32", description="Model precision")
    optimize: bool = Field(default=True, description="Enable model optimization")
    cache_models: bool = Field(default=True, description="Cache loaded models")
    model_timeout: float = Field(default=60.0, ge=0, description="Model loading timeout")
    warmup_requests: int = Field(default=5, ge=0, description="Number of warmup requests")
    
    @field_validator('device')
    @classmethod
    def validate_device(cls, v):
        """Validate device specification."""
        if v not in ["auto", "cpu", "cuda", "cuda:0", "cuda:1"]:
            raise ValueError("Device must be 'auto', 'cpu', 'cuda', 'cuda:0', or 'cuda:1'")
        return v
    
    @field_validator('precision')
    @classmethod
    def validate_precision(cls, v):
        """Validate precision specification."""
        if v not in ["float32", "float16", "int8"]:
            raise ValueError("Precision must be 'float32', 'float16', or 'int8'")
        return v
    
    @model_validator(mode='after')
    def validate_batch_sizes(self):
        """Validate batch size configuration."""
        if self.max_batch_size < self.batch_size:
            raise ValueError("max_batch_size must be >= batch_size")
        return self


class LoggingConfig(BaseModel):
    """Logging configuration."""
    
    model_config = ConfigDict(extra="forbid")
    
    level: LogLevel = Field(default=LogLevel.INFO, description="Log level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    date_format: str = Field(default="%Y-%m-%d %H:%M:%S", description="Date format")
    file_path: Optional[FilePath] = Field(default=None, description="Log file path")
    max_file_size: int = Field(default=10 * 1024 * 1024, ge=1, description="Max log file size")
    backup_count: int = Field(default=5, ge=0, description="Number of backup files")
    json_format: bool = Field(default=False, description="Use JSON format")
    include_timestamp: bool = Field(default=True, description="Include timestamp")
    include_level: bool = Field(default=True, description="Include log level")
    include_logger: bool = Field(default=True, description="Include logger name")
    correlation_id: bool = Field(default=True, description="Include correlation ID")


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""
    
    model_config = ConfigDict(extra="forbid")
    
    enabled: bool = Field(default=True, description="Enable monitoring")
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    tracing_enabled: bool = Field(default=True, description="Enable tracing")
    health_check_enabled: bool = Field(default=True, description="Enable health checks")
    prometheus_enabled: bool = Field(default=True, description="Enable Prometheus metrics")
    jaeger_enabled: bool = Field(default=False, description="Enable Jaeger tracing")
    metrics_port: int = Field(default=9090, ge=1, le=65535, description="Metrics port")
    metrics_path: str = Field(default="/metrics", description="Metrics endpoint path")
    collect_system_metrics: bool = Field(default=True, description="Collect system metrics")
    collect_model_metrics: bool = Field(default=True, description="Collect model metrics")
    jaeger_endpoint: Optional[HttpUrl] = Field(default=None, description="Jaeger endpoint")
    trace_sample_rate: float = Field(default=0.1, ge=0, le=1, description="Trace sampling rate")
    health_check_path: str = Field(default="/health", description="Health check endpoint")
    health_check_interval: float = Field(default=30.0, ge=0, description="Health check interval")
    readiness_probe_path: str = Field(default="/ready", description="Readiness probe endpoint")
    liveness_probe_path: str = Field(default="/live", description="Liveness probe endpoint")


class DatabaseConfig(BaseModel):
    """Database configuration."""
    
    model_config = ConfigDict(extra="forbid")
    
    backend: DatabaseBackend = Field(default=DatabaseBackend.SQLITE, description="Database backend")
    host: Optional[str] = Field(default=None, description="Database host")
    port: Optional[int] = Field(default=None, ge=1, le=65535, description="Database port")
    name: str = Field(default="model_serving", description="Database name")
    username: Optional[str] = Field(default=None, description="Database username")
    password: Optional[SecretStr] = Field(default=None, description="Database password")
    url: Optional[str] = Field(default=None, description="Database connection URL")
    pool_size: int = Field(default=5, ge=1, description="Connection pool size")
    max_overflow: int = Field(default=10, ge=0, description="Max pool overflow")
    pool_timeout: float = Field(default=30.0, ge=0, description="Pool timeout")
    pool_recycle: int = Field(default=3600, ge=0, description="Pool recycle time")
    echo: bool = Field(default=False, description="Echo SQL queries")
    
    def get_database_url(self) -> str:
        """Get database connection URL."""
        if self.url:
            return self.url
        
        if self.backend == DatabaseBackend.SQLITE:
            return f"sqlite:///{self.name}.db"
        elif self.backend == DatabaseBackend.POSTGRESQL:
            host = self.host or "localhost"
            port = self.port or 5432
            username = self.username or "postgres"
            password = self.password.get_secret_value() if self.password else ""
            return f"postgresql://{username}:{password}@{host}:{port}/{self.name}"
        elif self.backend == DatabaseBackend.MYSQL:
            host = self.host or "localhost"
            port = self.port or 3306
            username = self.username or "root"
            password = self.password.get_secret_value() if self.password else ""
            return f"mysql://{username}:{password}@{host}:{port}/{self.name}"
        else:
            raise ValueError(f"Unsupported database backend: {self.backend}")
    
    @model_validator(mode='after')
    def validate_database_config(self):
        """Validate database configuration."""
        if self.backend in [DatabaseBackend.POSTGRESQL, DatabaseBackend.MYSQL]:
            if not self.host and not self.url:
                raise ValueError(f"Host or URL required for {self.backend.value} backend")
            if not self.username:
                raise ValueError(f"Username required for {self.backend.value} backend")
        return self


class CacheConfig(BaseModel):
    """Cache configuration."""
    
    model_config = ConfigDict(extra="forbid")
    
    backend: CacheBackend = Field(default=CacheBackend.MEMORY, description="Cache backend")
    host: Optional[str] = Field(default=None, description="Cache host")
    port: Optional[int] = Field(default=None, ge=1, le=65535, description="Cache port")
    password: Optional[SecretStr] = Field(default=None, description="Cache password")
    db: int = Field(default=0, ge=0, description="Cache database number")
    max_connections: int = Field(default=10, ge=1, description="Max connections")
    timeout: float = Field(default=5.0, ge=0, description="Connection timeout")
    key_prefix: str = Field(default="model_serving:", description="Key prefix")
    default_ttl: int = Field(default=3600, ge=0, description="Default TTL in seconds")
    max_memory: Optional[int] = Field(default=None, ge=1, description="Max memory usage")
    
    def get_cache_url(self) -> str:
        """Get cache connection URL."""
        if self.backend == CacheBackend.MEMORY:
            return "memory://"
        elif self.backend == CacheBackend.REDIS:
            host = self.host or "localhost"
            port = self.port or 6379
            password = f":{self.password.get_secret_value()}" if self.password else ""
            return f"redis://{password}@{host}:{port}/{self.db}"
        elif self.backend == CacheBackend.MEMCACHED:
            host = self.host or "localhost"
            port = self.port or 11211
            return f"memcached://{host}:{port}"
        else:
            raise ValueError(f"Unsupported cache backend: {self.backend}")
    
    @model_validator(mode='after')
    def validate_cache_backend(self):
        """Validate cache backend configuration."""
        if self.backend in [CacheBackend.REDIS, CacheBackend.MEMCACHED]:
            if not self.host:
                raise ValueError(f"Host required for {self.backend.value} backend")
        return self


class SecurityConfig(BaseModel):
    """Security configuration."""
    
    model_config = ConfigDict(extra="forbid")
    
    enabled: bool = Field(default=False, description="Enable security features")
    api_key_required: bool = Field(default=False, description="Require API key")
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    api_keys: List[SecretStr] = Field(default_factory=list, description="Valid API keys")
    cors_enabled: bool = Field(default=True, description="Enable CORS")
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")
    cors_methods: List[str] = Field(default=["GET", "POST", "PUT", "DELETE"], description="CORS allowed methods")
    cors_headers: List[str] = Field(default=["*"], description="CORS allowed headers")
    rate_limit_enabled: bool = Field(default=False, description="Enable rate limiting")
    rate_limit_requests: int = Field(default=100, ge=1, description="Rate limit requests per window")
    rate_limit_window: int = Field(default=60, ge=1, description="Rate limit window in seconds")
    ssl_enabled: bool = Field(default=False, description="Enable SSL")
    ssl_cert_file: Optional[FilePath] = Field(default=None, description="SSL certificate file")
    ssl_key_file: Optional[FilePath] = Field(default=None, description="SSL private key file")
    
    @model_validator(mode='after')
    def validate_ssl_files(self):
        """Validate SSL configuration."""
        if self.ssl_enabled:
            if not self.ssl_cert_file or not self.ssl_key_file:
                raise ValueError("SSL certificate and key files required when SSL is enabled")
        return self


class AppConfig(BaseConfig):
    """Main application configuration."""
    
    model_config = ConfigDict(extra="forbid")
    
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Application environment")
    debug: bool = Field(default=False, description="Debug mode")
    version: str = Field(default="1.0.0", description="Application version")
    name: str = Field(default="Model Serving", description="Application name")
    description: str = Field(default="High-performance model serving API", description="Application description")
    
    # Sub-configurations
    server: ServerConfig = Field(default_factory=ServerConfig, description="Server configuration")
    model: ModelConfig = Field(default_factory=ModelConfig, description="Model configuration")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig, description="Monitoring configuration")
    database: DatabaseConfig = Field(default_factory=DatabaseConfig, description="Database configuration")
    cache: CacheConfig = Field(default_factory=CacheConfig, description="Cache configuration")
    security: SecurityConfig = Field(default_factory=SecurityConfig, description="Security configuration")
    
    # Directory paths
    config_dir: Path = Field(default=Path("config"), description="Configuration directory")
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    log_dir: Path = Field(default=Path("logs"), description="Log directory")
    temp_dir: Path = Field(default=Path("temp"), description="Temporary directory")
    
    @model_validator(mode='after')
    def ensure_directories_exist(self):
        """Ensure required directories exist."""
        for directory in [self.config_dir, self.data_dir, self.log_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        return self
    
    @model_validator(mode='after')
    def validate_production_config(self):
        """Validate production-specific configuration."""
        if self.environment == Environment.PRODUCTION:
            if self.debug:
                raise ValueError("Debug mode should not be enabled in production")
            if self.server.reload:
                raise ValueError("Auto-reload should not be enabled in production")
            if not self.security.enabled:
                raise ValueError("Security must be enabled in production")
        return self