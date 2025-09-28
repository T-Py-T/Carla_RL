# CarlaRL Production Monitoring Guide

This guide covers the comprehensive monitoring and observability system for the CarlaRL Policy-as-a-Service infrastructure.

## Overview

The monitoring system provides:

- **Prometheus Metrics**: Comprehensive metrics collection for performance, errors, and system resources
- **Structured Logging**: JSON-formatted logs with correlation IDs for request tracing
- **Health Checks**: Detailed health status with system resource monitoring
- **Distributed Tracing**: Request flow analysis and performance tracking
- **Grafana Dashboards**: Real-time visualization of metrics and performance
- **Alerting**: Automated alerts for performance degradation and errors
- **Log Analysis**: Tools for aggregating and analyzing logs

## Quick Start

### 1. Start the Monitoring Stack

```bash
cd deploy
./scripts/setup_monitoring.sh
```

This will start:
- Prometheus (http://localhost:9090)
- Grafana (http://localhost:3000, admin/admin)
- Alertmanager (http://localhost:9093)
- CarlaRL Service (http://localhost:8080)

### 2. View Metrics

- **Prometheus**: http://localhost:9090
- **Grafana Dashboard**: http://localhost:3000 (auto-imported)
- **Service Metrics**: http://localhost:8080/metrics

### 3. Analyze Logs

```bash
# Analyze logs from file
python scripts/log_analyzer.py --file logs/app.log --summary

# Analyze logs from stdin
tail -f logs/app.log | python scripts/log_analyzer.py --level ERROR

# Export analysis to JSON
python scripts/log_analyzer.py --file logs/app.log --output analysis.json
```

## Monitoring Components

### 1. Prometheus Metrics

The service exposes comprehensive metrics at `/metrics`:

#### Inference Metrics
- `carla_rl_inference_duration_seconds` - Inference latency histogram
- `carla_rl_inference_requests_total` - Total inference requests
- `carla_rl_batch_size` - Batch size distribution
- `carla_rl_deterministic_requests_total` - Deterministic vs non-deterministic

#### System Metrics
- `carla_rl_cpu_usage_percent` - CPU utilization
- `carla_rl_memory_usage_bytes` - Memory usage
- `carla_rl_gpu_utilization_percent` - GPU utilization
- `carla_rl_gpu_memory_usage_bytes` - GPU memory usage

#### Error Metrics
- `carla_rl_errors_total` - Error counts by type
- `carla_rl_http_requests_total` - HTTP request counts
- `carla_rl_request_duration_seconds` - Request duration histogram

#### Model Metrics
- `carla_rl_model_loaded` - Model loading status
- `carla_rl_model_warmed_up` - Model warmup status
- `carla_rl_model_loading_duration_seconds` - Model loading time
- `carla_rl_model_warmup_duration_seconds` - Model warmup time

### 2. Structured Logging

All logs are in JSON format with correlation IDs:

```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "INFO",
  "logger": "carla_rl_server",
  "correlation_id": "req-12345",
  "event_type": "inference",
  "message": "Inference request completed",
  "model_version": "v1.0.0",
  "device": "cpu",
  "batch_size": 4,
  "duration_ms": 25.5,
  "status": "success"
}
```

#### Log Levels
- `DEBUG`: Detailed debugging information
- `INFO`: General information about operations
- `WARNING`: Warning conditions
- `ERROR`: Error conditions
- `CRITICAL`: Critical error conditions

#### Event Types
- `startup`: Service startup events
- `model_loaded`: Model loading events
- `model_warmup`: Model warmup events
- `inference`: Inference request events
- `http_request`: HTTP request events
- `health_check`: Health check events
- `error`: Error events
- `shutdown`: Service shutdown events

### 3. Health Checks

The `/healthz` endpoint provides comprehensive health status:

```json
{
  "status": "healthy",
  "version": "v1.0.0",
  "git": "abc123",
  "device": "cpu"
}
```

#### Health Check Components
- **Model Status**: Loaded and warmed up status
- **System Resources**: CPU, memory, disk usage
- **GPU Status**: GPU availability and utilization
- **Service Uptime**: Service running time
- **Dependencies**: External service health

### 4. Distributed Tracing

Request tracing provides end-to-end visibility:

#### Trace Components
- **Request Tracing**: HTTP request flow
- **Inference Tracing**: Model inference operations
- **Model Operations**: Loading and warmup
- **Health Checks**: Health check operations

#### Trace Information
- Trace ID and Span ID
- Operation name and duration
- Tags and metadata
- Error information
- Parent-child relationships

### 5. Grafana Dashboards

Pre-configured dashboards show:

#### Service Overview
- Model loading status
- Service uptime
- Overall health

#### Performance Metrics
- Inference latency (P50, P95, P99)
- Throughput (requests/second)
- Error rates
- Request duration by endpoint

#### System Resources
- CPU and memory usage
- GPU utilization
- Disk space

#### Model Performance
- Performance by version
- Batch size distribution
- Deterministic vs non-deterministic

### 6. Alerting

Automated alerts for:

#### Performance Alerts
- High inference latency (>50ms P95)
- Critical latency (>100ms P95)
- Low throughput
- High request duration

#### Error Alerts
- High error rate (>0.1 errors/sec)
- Critical error rate (>1.0 errors/sec)
- Model not loaded
- Service down

#### Resource Alerts
- High CPU usage (>80%)
- High memory usage (>85%)
- High GPU utilization (>90%)
- High GPU memory usage

## Configuration

### Environment Variables

```bash
# Logging
LOG_LEVEL=INFO                    # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

# Monitoring
ENABLE_METRICS=true              # Enable metrics collection
ENABLE_TRACING=true              # Enable distributed tracing
ENABLE_HEALTH_CHECKS=true        # Enable health checks

# Prometheus
PROMETHEUS_PORT=9090             # Prometheus port
METRICS_PATH=/metrics            # Metrics endpoint path

# Grafana
GRAFANA_PORT=3000                # Grafana port
GRAFANA_ADMIN_PASSWORD=admin     # Grafana admin password
```

### Prometheus Configuration

Edit `deploy/prometheus/prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'carla-rl-serving'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 5s
```

### Grafana Dashboard

Import `deploy/grafana/carla-rl-dashboard.json` or use the auto-import feature.

### Alert Rules

Configure alerts in `deploy/prometheus/carla_rl_alerts.yml`:

```yaml
groups:
  - name: carla_rl_serving
    rules:
      - alert: HighInferenceLatency
        expr: histogram_quantile(0.95, rate(carla_rl_inference_duration_seconds_bucket[5m])) > 0.05
        for: 2m
        labels:
          severity: warning
```

## Log Analysis

### Command Line Usage

```bash
# Basic analysis
python scripts/log_analyzer.py --file logs/app.log --summary

# Filter by level
python scripts/log_analyzer.py --file logs/app.log --level ERROR

# Filter by event type
python scripts/log_analyzer.py --file logs/app.log --event-type inference

# Search by pattern
python scripts/log_analyzer.py --file logs/app.log --search "timeout"

# Filter by time range
python scripts/log_analyzer.py --file logs/app.log --time-start "2024-01-15T10:00:00" --time-end "2024-01-15T11:00:00"

# Filter by correlation ID
python scripts/log_analyzer.py --file logs/app.log --correlation-id "req-12345"

# Export to JSON
python scripts/log_analyzer.py --file logs/app.log --output analysis.json
```

### Programmatic Usage

```python
from src.monitoring.log_aggregation import LogAggregator

# Create aggregator
aggregator = LogAggregator()

# Load logs
aggregator.load_from_file('logs/app.log')

# Get performance summary
summary = aggregator.get_performance_summary()
print(f"Inference P95: {summary['inference']['p95']}ms")

# Get error analysis
errors = aggregator.get_error_analysis()
print(f"Total errors: {errors['total_errors']}")

# Get correlation trace
trace = aggregator.get_correlation_trace('req-12345')
for entry in trace:
    print(f"{entry.timestamp}: {entry.message}")
```

## Troubleshooting

### Common Issues

#### 1. Metrics Not Appearing
- Check that the service is running
- Verify `/metrics` endpoint is accessible
- Check Prometheus configuration
- Ensure metrics collection is enabled

#### 2. Logs Not Structured
- Verify `LOG_LEVEL` environment variable
- Check that monitoring is initialized
- Ensure JSON formatter is configured

#### 3. Health Checks Failing
- Check model loading status
- Verify system resources
- Check GPU availability
- Review health check logs

#### 4. Alerts Not Firing
- Verify Prometheus is scraping metrics
- Check alert rule configuration
- Ensure Alertmanager is running
- Check alert thresholds

#### 5. Dashboard Not Loading
- Verify Grafana is running
- Check dashboard import
- Ensure Prometheus data source is configured
- Check time range settings

### Debug Commands

```bash
# Check service status
curl http://localhost:8080/healthz

# Check metrics
curl http://localhost:8080/metrics

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Check Grafana health
curl http://localhost:3000/api/health

# Analyze recent logs
tail -100 logs/app.log | python scripts/log_analyzer.py --summary
```

## Performance Considerations

### Metrics Collection
- Metrics are collected in background threads
- Minimal performance impact on inference
- Configurable collection intervals

### Logging
- Structured logging has minimal overhead
- Correlation IDs are thread-safe
- Log levels can be adjusted for performance

### Health Checks
- Health checks run on-demand
- Cached results for performance
- Configurable check intervals

### Tracing
- Tracing spans are lightweight
- Configurable trace retention
- Optional sampling for high-throughput scenarios

## Security Considerations

### Metrics
- Metrics endpoint should be protected in production
- Consider authentication for Prometheus
- Limit access to sensitive metrics

### Logs
- Logs may contain sensitive information
- Implement log rotation and retention
- Consider log encryption for sensitive data

### Dashboards
- Secure Grafana access
- Use strong passwords
- Implement proper user management

### Alerts
- Secure Alertmanager configuration
- Use encrypted channels for notifications
- Implement proper alert routing

## Best Practices

### Monitoring
1. Set up comprehensive alerting
2. Monitor key performance indicators
3. Use appropriate alert thresholds
4. Implement alert escalation procedures

### Logging
1. Use structured logging consistently
2. Include correlation IDs for tracing
3. Log at appropriate levels
4. Implement log rotation

### Health Checks
1. Check all critical dependencies
2. Use appropriate health check intervals
3. Implement graceful degradation
4. Monitor health check performance

### Dashboards
1. Keep dashboards focused and relevant
2. Use appropriate time ranges
3. Implement dashboard versioning
4. Regular dashboard maintenance

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review log files for errors
3. Check Prometheus and Grafana status
4. Verify configuration settings
5. Contact the development team

## Changelog

### v1.0.0
- Initial monitoring system implementation
- Prometheus metrics collection
- Structured JSON logging
- Health check system
- Distributed tracing
- Grafana dashboards
- Alerting system
- Log analysis tools
