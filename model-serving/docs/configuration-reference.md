# Configuration Reference

This document provides a comprehensive reference for all configuration options available in the CarlaRL Policy-as-a-Service system.

## Table of Contents

- [Overview](#overview)
- [Configuration Sources](#configuration-sources)
- [Environment Variables](#environment-variables)
- [Configuration Files](#configuration-files)
- [Configuration Sections](#configuration-sections)
- [Validation Rules](#validation-rules)
- [Examples](#examples)
- [Best Practices](#best-practices)

## Overview

The configuration system uses Pydantic models for type-safe configuration with comprehensive validation. Configuration can be loaded from multiple sources with a clear precedence order.

### Configuration Precedence

1. **Environment Variables** (highest priority)
2. **Configuration Files** (YAML/JSON)
3. **Default Values** (lowest priority)

### Supported Formats

- **YAML**: `.yaml`, `.yml`
- **JSON**: `.json`
- **Environment Variables**: `.env` files
- **Command Line**: Via CLI arguments

## Configuration Sources

### 1. Environment Variables

Environment variables are automatically mapped to configuration fields using the pattern `{SECTION}_{FIELD}`.

```bash
# Server configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8080
SERVER_WORKERS=4

# Model configuration
MODEL_BACKEND=pytorch
MODEL_DEVICE=cuda
MODEL_BATCH_SIZE=1

# Logging configuration
LOGGING_LEVEL=INFO
LOGGING_JSON_FORMAT=true
```

### 2. Configuration Files

Configuration files are loaded from the `config/` directory by default.

```yaml
# config/production.yaml
server:
  host: "0.0.0.0"
  port: 8080
  workers: 4

model:
  backend: "pytorch"
  device: "cuda"
  batch_size: 1

logging:
  level: "INFO"
  json_format: true
```

### 3. Environment-Specific Files

The system automatically loads environment-specific configuration files:

- `config/development.yaml`
- `config/staging.yaml`
- `config/production.yaml`
- `config/testing.yaml`

## Environment Variables

### Server Configuration

| Variable | Default | Description | Type |
|----------|---------|-------------|------|
| `SERVER_HOST` | `0.0.0.0` | Server host address | IPvAnyAddress |
| `SERVER_PORT` | `8000` | Server port | int (1-65535) |
| `SERVER_WORKERS` | `1` | Number of worker processes | PositiveInt |
| `SERVER_RELOAD` | `false` | Enable auto-reload in development | bool |
| `SERVER_LOG_LEVEL` | `INFO` | Logging level | LogLevel |
| `SERVER_ACCESS_LOG` | `true` | Enable access logging | bool |
| `SERVER_MAX_REQUEST_SIZE` | `16777216` | Maximum request size in bytes | PositiveInt |
| `SERVER_TIMEOUT` | `30.0` | Request timeout in seconds | PositiveFloat |
| `SERVER_KEEP_ALIVE` | `true` | Enable keep-alive connections | bool |
| `SERVER_KEEP_ALIVE_TIMEOUT` | `5.0` | Keep-alive timeout in seconds | PositiveFloat |
| `SERVER_MAX_CONNECTIONS` | `1000` | Maximum concurrent connections | PositiveInt |

### Model Configuration

| Variable | Default | Description | Type |
|----------|---------|-------------|------|
| `MODEL_BACKEND` | `pytorch` | Model backend | ModelBackend |
| `MODEL_MODEL_PATH` | `null` | Path to model directory | DirectoryPath |
| `MODEL_MODEL_FILE` | `null` | Specific model file | FilePath |
| `MODEL_CONFIG_FILE` | `null` | Model configuration file | FilePath |
| `MODEL_DEVICE` | `auto` | Device to run model on | str |
| `MODEL_BATCH_SIZE` | `1` | Default batch size | PositiveInt |
| `MODEL_MAX_BATCH_SIZE` | `32` | Maximum batch size | PositiveInt |
| `MODEL_PRECISION` | `float32` | Model precision | Literal |
| `MODEL_OPTIMIZE` | `true` | Enable model optimization | bool |
| `MODEL_CACHE_MODELS` | `true` | Cache loaded models in memory | bool |
| `MODEL_MODEL_TIMEOUT` | `60.0` | Model loading timeout in seconds | PositiveFloat |
| `MODEL_WARMUP_REQUESTS` | `5` | Number of warmup requests | NonNegativeInt |

### Logging Configuration

| Variable | Default | Description | Type |
|----------|---------|-------------|------|
| `LOGGING_LEVEL` | `INFO` | Logging level | LogLevel |
| `LOGGING_FORMAT` | `%(asctime)s - %(name)s - %(levelname)s - %(message)s` | Log format string | str |
| `LOGGING_DATE_FORMAT` | `%Y-%m-%d %H:%M:%S` | Date format in logs | str |
| `LOGGING_FILE_PATH` | `null` | Log file path | FilePath |
| `LOGGING_MAX_FILE_SIZE` | `10485760` | Maximum log file size in bytes | PositiveInt |
| `LOGGING_BACKUP_COUNT` | `5` | Number of backup log files | NonNegativeInt |
| `LOGGING_JSON_FORMAT` | `false` | Use JSON log format | bool |
| `LOGGING_INCLUDE_TIMESTAMP` | `true` | Include timestamp in logs | bool |
| `LOGGING_INCLUDE_LEVEL` | `true` | Include level in logs | bool |
| `LOGGING_INCLUDE_LOGGER` | `true` | Include logger name in logs | bool |
| `LOGGING_CORRELATION_ID` | `true` | Include correlation ID in logs | bool |

### Monitoring Configuration

| Variable | Default | Description | Type |
|----------|---------|-------------|------|
| `MONITORING_ENABLED` | `true` | Enable monitoring | bool |
| `MONITORING_METRICS_ENABLED` | `true` | Enable metrics collection | bool |
| `MONITORING_TRACING_ENABLED` | `true` | Enable distributed tracing | bool |
| `MONITORING_HEALTH_CHECK_ENABLED` | `true` | Enable health checks | bool |
| `MONITORING_PROMETHEUS_ENABLED` | `true` | Enable Prometheus metrics | bool |
| `MONITORING_JAEGER_ENABLED` | `false` | Enable Jaeger tracing | bool |
| `MONITORING_METRICS_PORT` | `9090` | Prometheus metrics port | int (1-65535) |
| `MONITORING_METRICS_PATH` | `/metrics` | Metrics endpoint path | str |
| `MONITORING_COLLECT_SYSTEM_METRICS` | `true` | Collect system-level metrics | bool |
| `MONITORING_COLLECT_MODEL_METRICS` | `true` | Collect model-specific metrics | bool |
| `MONITORING_JAEGER_ENDPOINT` | `null` | Jaeger collector endpoint | HttpUrl |
| `MONITORING_TRACE_SAMPLE_RATE` | `0.1` | Trace sampling rate | float (0.0-1.0) |
| `MONITORING_HEALTH_CHECK_PATH` | `/health` | Health check endpoint | str |
| `MONITORING_HEALTH_CHECK_INTERVAL` | `30.0` | Health check interval in seconds | PositiveFloat |
| `MONITORING_READINESS_PROBE_PATH` | `/ready` | Readiness probe endpoint | str |
| `MONITORING_LIVENESS_PROBE_PATH` | `/live` | Liveness probe endpoint | str |

### Database Configuration

| Variable | Default | Description | Type |
|----------|---------|-------------|------|
| `DATABASE_BACKEND` | `sqlite` | Database backend | DatabaseBackend |
| `DATABASE_HOST` | `null` | Database host | str |
| `DATABASE_PORT` | `null` | Database port | int (1-65535) |
| `DATABASE_NAME` | `model_serving` | Database name | str |
| `DATABASE_USERNAME` | `null` | Database username | str |
| `DATABASE_PASSWORD` | `null` | Database password | SecretStr |
| `DATABASE_URL` | `null` | Database connection URL | str |
| `DATABASE_POOL_SIZE` | `5` | Connection pool size | PositiveInt |
| `DATABASE_MAX_OVERFLOW` | `10` | Maximum pool overflow | NonNegativeInt |
| `DATABASE_POOL_TIMEOUT` | `30.0` | Pool timeout in seconds | PositiveFloat |
| `DATABASE_POOL_RECYCLE` | `3600` | Pool recycle time in seconds | PositiveInt |
| `DATABASE_ECHO` | `false` | Echo SQL queries | bool |

### Cache Configuration

| Variable | Default | Description | Type |
|----------|---------|-------------|------|
| `CACHE_BACKEND` | `memory` | Cache backend | CacheBackend |
| `CACHE_HOST` | `null` | Cache host | str |
| `CACHE_PORT` | `null` | Cache port | int (1-65535) |
| `CACHE_PASSWORD` | `null` | Cache password | SecretStr |
| `CACHE_DB` | `0` | Cache database number | NonNegativeInt |
| `CACHE_MAX_CONNECTIONS` | `10` | Maximum cache connections | PositiveInt |
| `CACHE_TIMEOUT` | `5.0` | Cache timeout in seconds | PositiveFloat |
| `CACHE_KEY_PREFIX` | `model_serving:` | Cache key prefix | str |
| `CACHE_DEFAULT_TTL` | `3600` | Default TTL in seconds | PositiveInt |
| `CACHE_MAX_MEMORY` | `null` | Maximum memory usage | str |

### Security Configuration

| Variable | Default | Description | Type |
|----------|---------|-------------|------|
| `SECURITY_ENABLED` | `false` | Enable security features | bool |
| `SECURITY_API_KEY_REQUIRED` | `false` | Require API key | bool |
| `SECURITY_API_KEY_HEADER` | `X-API-Key` | API key header name | str |
| `SECURITY_API_KEYS` | `[]` | Valid API keys | List[SecretStr] |
| `SECURITY_CORS_ENABLED` | `true` | Enable CORS | bool |
| `SECURITY_CORS_ORIGINS` | `["*"]` | Allowed CORS origins | List[str] |
| `SECURITY_CORS_METHODS` | `["GET", "POST", "PUT", "DELETE"]` | Allowed CORS methods | List[str] |
| `SECURITY_CORS_HEADERS` | `["*"]` | Allowed CORS headers | List[str] |
| `SECURITY_RATE_LIMIT_ENABLED` | `false` | Enable rate limiting | bool |
| `SECURITY_RATE_LIMIT_REQUESTS` | `100` | Rate limit requests per minute | PositiveInt |
| `SECURITY_RATE_LIMIT_WINDOW` | `60` | Rate limit window in seconds | PositiveInt |
| `SECURITY_SSL_ENABLED` | `false` | Enable SSL/TLS | bool |
| `SECURITY_SSL_CERT_FILE` | `null` | SSL certificate file | FilePath |
| `SECURITY_SSL_KEY_FILE` | `null` | SSL private key file | FilePath |

### Application Configuration

| Variable | Default | Description | Type |
|----------|---------|-------------|------|
| `ENVIRONMENT` | `development` | Application environment | Environment |
| `DEBUG` | `false` | Debug mode | bool |
| `VERSION` | `1.0.0` | Application version | str |
| `NAME` | `Model Serving` | Application name | str |
| `DESCRIPTION` | `High-performance model serving API` | Application description | str |
| `CONFIG_DIR` | `config` | Configuration directory | Path |
| `DATA_DIR` | `data` | Data directory | Path |
| `LOG_DIR` | `logs` | Log directory | Path |
| `TEMP_DIR` | `temp` | Temporary directory | Path |

## Configuration Sections

### Server Configuration

```yaml
server:
  host: "0.0.0.0"                    # Server host address
  port: 8080                         # Server port
  workers: 4                         # Number of worker processes
  reload: false                      # Enable auto-reload in development
  log_level: "INFO"                  # Logging level
  access_log: true                   # Enable access logging
  max_request_size: 16777216         # Maximum request size in bytes
  timeout: 30.0                      # Request timeout in seconds
  keep_alive: true                   # Enable keep-alive connections
  keep_alive_timeout: 5.0            # Keep-alive timeout in seconds
  max_connections: 1000              # Maximum concurrent connections
```

### Model Configuration

```yaml
model:
  backend: "pytorch"                 # Model backend (pytorch, tensorflow, onnx, torchscript, triton)
  model_path: "/path/to/models"      # Path to model directory
  model_file: "/path/to/model.pt"    # Specific model file
  config_file: "/path/to/config.yaml" # Model configuration file
  device: "auto"                     # Device to run model on (cpu, cuda, auto, cuda:0, etc.)
  batch_size: 1                      # Default batch size
  max_batch_size: 32                 # Maximum batch size
  precision: "float32"               # Model precision (float32, float16, int8)
  optimize: true                     # Enable model optimization
  cache_models: true                 # Cache loaded models in memory
  model_timeout: 60.0                # Model loading timeout in seconds
  warmup_requests: 5                 # Number of warmup requests
```

### Logging Configuration

```yaml
logging:
  level: "INFO"                      # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s" # Log format string
  date_format: "%Y-%m-%d %H:%M:%S"  # Date format in logs
  file_path: "/path/to/logs/app.log" # Log file path
  max_file_size: 10485760            # Maximum log file size in bytes
  backup_count: 5                    # Number of backup log files
  json_format: false                 # Use JSON log format
  include_timestamp: true            # Include timestamp in logs
  include_level: true                # Include level in logs
  include_logger: true               # Include logger name in logs
  correlation_id: true               # Include correlation ID in logs
```

### Monitoring Configuration

```yaml
monitoring:
  enabled: true                      # Enable monitoring
  metrics_enabled: true              # Enable metrics collection
  tracing_enabled: true              # Enable distributed tracing
  health_check_enabled: true         # Enable health checks
  prometheus_enabled: true           # Enable Prometheus metrics
  jaeger_enabled: false              # Enable Jaeger tracing
  
  # Metrics configuration
  metrics_port: 9090                 # Prometheus metrics port
  metrics_path: "/metrics"           # Metrics endpoint path
  collect_system_metrics: true       # Collect system-level metrics
  collect_model_metrics: true        # Collect model-specific metrics
  
  # Tracing configuration
  jaeger_endpoint: "http://jaeger:14268/api/traces" # Jaeger collector endpoint
  trace_sample_rate: 0.1             # Trace sampling rate (0.0-1.0)
  
  # Health check configuration
  health_check_path: "/health"       # Health check endpoint
  health_check_interval: 30.0        # Health check interval in seconds
  readiness_probe_path: "/ready"     # Readiness probe endpoint
  liveness_probe_path: "/live"       # Liveness probe endpoint
```

### Database Configuration

```yaml
database:
  backend: "sqlite"                  # Database backend (postgresql, mysql, sqlite, mongodb)
  host: "localhost"                  # Database host
  port: 5432                         # Database port
  name: "model_serving"              # Database name
  username: "user"                   # Database username
  password: "password"               # Database password
  url: "postgresql://user:password@localhost:5432/model_serving" # Database connection URL
  pool_size: 5                       # Connection pool size
  max_overflow: 10                   # Maximum pool overflow
  pool_timeout: 30.0                 # Pool timeout in seconds
  pool_recycle: 3600                 # Pool recycle time in seconds
  echo: false                        # Echo SQL queries
```

### Cache Configuration

```yaml
cache:
  backend: "redis"                   # Cache backend (redis, memcached, memory, disk)
  host: "localhost"                  # Cache host
  port: 6379                         # Cache port
  password: "password"               # Cache password
  db: 0                              # Cache database number
  max_connections: 10                # Maximum cache connections
  timeout: 5.0                       # Cache timeout in seconds
  key_prefix: "model_serving:"       # Cache key prefix
  default_ttl: 3600                  # Default TTL in seconds
  max_memory: "100MB"                # Maximum memory usage
```

### Security Configuration

```yaml
security:
  enabled: false                     # Enable security features
  api_key_required: false            # Require API key
  api_key_header: "X-API-Key"        # API key header name
  api_keys:                          # Valid API keys
    - "secret-key-1"
    - "secret-key-2"
  cors_enabled: true                 # Enable CORS
  cors_origins:                      # Allowed CORS origins
    - "https://yourdomain.com"
    - "https://api.yourdomain.com"
  cors_methods:                      # Allowed CORS methods
    - "GET"
    - "POST"
    - "PUT"
    - "DELETE"
  cors_headers:                      # Allowed CORS headers
    - "Content-Type"
    - "Authorization"
  rate_limit_enabled: false          # Enable rate limiting
  rate_limit_requests: 100           # Rate limit requests per minute
  rate_limit_window: 60              # Rate limit window in seconds
  ssl_enabled: false                 # Enable SSL/TLS
  ssl_cert_file: "/path/to/cert.pem" # SSL certificate file
  ssl_key_file: "/path/to/key.pem"   # SSL private key file
```

## Validation Rules

### Server Configuration

- **Port**: Must be between 1 and 65535
- **Workers**: Must be positive integer
- **Max Request Size**: Must be positive integer
- **Timeout**: Must be positive float
- **Keep Alive Timeout**: Must be positive float
- **Max Connections**: Must be positive integer
- **Workers with Reload**: Cannot use multiple workers with reload enabled

### Model Configuration

- **Device**: Must be valid device specification (cpu, cuda, auto, cuda:0, etc.)
- **Max Batch Size**: Must be >= batch_size
- **Precision**: Must be one of float32, float16, int8
- **Backend**: Must be valid backend type

### Database Configuration

- **Host/URL**: Required for PostgreSQL, MySQL, MongoDB backends
- **Port**: Must be between 1 and 65535
- **Pool Size**: Must be positive integer
- **Max Overflow**: Must be non-negative integer
- **Pool Timeout**: Must be positive float
- **Pool Recycle**: Must be positive integer

### Cache Configuration

- **Host**: Required for Redis and Memcached backends
- **Port**: Must be between 1 and 65535
- **Max Connections**: Must be positive integer
- **Timeout**: Must be positive float
- **Default TTL**: Must be positive integer

### Security Configuration

- **SSL Files**: Must exist when SSL is enabled
- **API Keys**: Must be non-empty when API key required
- **Rate Limit**: Must be positive integer
- **Rate Limit Window**: Must be positive integer

## Examples

### Development Configuration

```yaml
# config/development.yaml
environment: development
debug: true

server:
  host: "127.0.0.1"
  port: 8000
  workers: 1
  reload: true
  log_level: "DEBUG"

model:
  backend: "pytorch"
  device: "cpu"
  batch_size: 1
  optimize: false

logging:
  level: "DEBUG"
  json_format: false
  file_path: "logs/development.log"

monitoring:
  enabled: true
  metrics_enabled: true
  tracing_enabled: false

database:
  backend: "sqlite"
  name: "model_serving_dev"

cache:
  backend: "memory"

security:
  enabled: false
  cors_origins: ["*"]
```

### Production Configuration

```yaml
# config/production.yaml
environment: production
debug: false

server:
  host: "0.0.0.0"
  port: 8080
  workers: 4
  reload: false
  log_level: "INFO"
  max_request_size: 16777216
  timeout: 30.0
  keep_alive: true
  keep_alive_timeout: 5.0
  max_connections: 1000

model:
  backend: "pytorch"
  device: "cuda"
  batch_size: 4
  max_batch_size: 32
  precision: "float16"
  optimize: true
  cache_models: true

logging:
  level: "INFO"
  json_format: true
  file_path: "/var/log/carla-rl-serving/app.log"
  max_file_size: 104857600
  backup_count: 10

monitoring:
  enabled: true
  metrics_enabled: true
  tracing_enabled: true
  prometheus_enabled: true
  jaeger_enabled: true
  jaeger_endpoint: "http://jaeger:14268/api/traces"
  trace_sample_rate: 0.1

database:
  backend: "postgresql"
  host: "postgres"
  port: 5432
  name: "model_serving"
  username: "carla_rl"
  password: "secure_password"
  pool_size: 10
  max_overflow: 20

cache:
  backend: "redis"
  host: "redis"
  port: 6379
  password: "redis_password"
  max_connections: 20
  default_ttl: 3600

security:
  enabled: true
  api_key_required: true
  api_keys:
    - "prod-api-key-1"
    - "prod-api-key-2"
  cors_origins:
    - "https://yourdomain.com"
    - "https://api.yourdomain.com"
  rate_limit_enabled: true
  rate_limit_requests: 1000
  rate_limit_window: 60
  ssl_enabled: true
  ssl_cert_file: "/etc/ssl/certs/carla-rl-serving.crt"
  ssl_key_file: "/etc/ssl/private/carla-rl-serving.key"
```

### Docker Configuration

```yaml
# config/docker.yaml
environment: production
debug: false

server:
  host: "0.0.0.0"
  port: 8080
  workers: 2
  log_level: "INFO"

model:
  backend: "pytorch"
  device: "auto"
  batch_size: 1
  optimize: true

logging:
  level: "INFO"
  json_format: true

monitoring:
  enabled: true
  metrics_enabled: true
  tracing_enabled: false

database:
  backend: "sqlite"
  name: "/app/data/model_serving.db"

cache:
  backend: "memory"

security:
  enabled: false
  cors_origins: ["*"]
```

### Kubernetes Configuration

```yaml
# config/kubernetes.yaml
environment: production
debug: false

server:
  host: "0.0.0.0"
  port: 8080
  workers: 2
  log_level: "INFO"

model:
  backend: "pytorch"
  device: "auto"
  batch_size: 1
  optimize: true

logging:
  level: "INFO"
  json_format: true

monitoring:
  enabled: true
  metrics_enabled: true
  tracing_enabled: true
  prometheus_enabled: true
  jaeger_enabled: true
  jaeger_endpoint: "http://jaeger-collector:14268/api/traces"

database:
  backend: "postgresql"
  host: "postgres-service"
  port: 5432
  name: "model_serving"
  username: "carla_rl"
  password: "secure_password"

cache:
  backend: "redis"
  host: "redis-service"
  port: 6379
  password: "redis_password"

security:
  enabled: true
  api_key_required: true
  cors_origins:
    - "https://yourdomain.com"
  rate_limit_enabled: true
  rate_limit_requests: 1000
```

## Best Practices

### 1. Environment-Specific Configuration

- Use separate configuration files for each environment
- Keep sensitive data in environment variables or secrets
- Use configuration validation to catch errors early

### 2. Security

- Never commit secrets to version control
- Use environment variables for sensitive configuration
- Enable security features in production
- Use strong API keys and passwords

### 3. Performance

- Tune worker count based on CPU cores
- Use appropriate batch sizes for your hardware
- Enable model optimization in production
- Use caching for frequently accessed data

### 4. Monitoring

- Enable comprehensive monitoring in production
- Use structured logging (JSON format)
- Set up health checks and probes
- Monitor resource usage and performance

### 5. Database and Cache

- Use connection pooling for databases
- Configure appropriate timeouts
- Use persistent storage for production data
- Monitor cache hit rates

### 6. Logging

- Use appropriate log levels
- Enable structured logging in production
- Set up log rotation
- Include correlation IDs for tracing

## Next Steps

- [Docker Deployment Guide](deployment-guides/docker-deployment.md)
- [Kubernetes Deployment Guide](deployment-guides/kubernetes-deployment.md)
- [Performance Tuning Guide](performance-tuning/performance-tuning.md)
- [Monitoring Setup Guide](monitoring/monitoring-setup.md)
