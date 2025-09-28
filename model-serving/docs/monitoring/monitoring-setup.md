# Monitoring and Observability Setup Guide

This guide covers setting up comprehensive monitoring and observability for the CarlaRL Policy-as-a-Service system using Prometheus, Grafana, and other monitoring tools.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prometheus Setup](#prometheus-setup)
- [Grafana Setup](#grafana-setup)
- [Alerting Setup](#alerting-setup)
- [Logging Setup](#logging-setup)
- [Tracing Setup](#tracing-setup)
- [Custom Metrics](#custom-metrics)
- [Dashboards](#dashboards)
- [Troubleshooting](#troubleshooting)

## Overview

The monitoring stack provides comprehensive observability for:

- **Metrics**: Performance, resource usage, and business metrics
- **Logs**: Structured logging with correlation IDs
- **Traces**: Distributed tracing for request flow analysis
- **Alerts**: Proactive alerting for issues and anomalies
- **Dashboards**: Real-time visualization of system health

### Monitoring Components

- **Prometheus**: Metrics collection and storage
- **Grafana**: Metrics visualization and dashboards
- **AlertManager**: Alert routing and notification
- **Jaeger**: Distributed tracing
- **ELK Stack**: Log aggregation and analysis (optional)
- **Node Exporter**: System metrics collection

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Application   │────│   Prometheus     │────│     Grafana     │
│   (Metrics)     │    │   (Collection)   │    │  (Visualization)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   AlertManager  │    │   Node Exporter  │    │     Jaeger      │
│   (Alerts)      │    │  (System Metrics)│    │   (Tracing)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Prometheus Setup

### 1. Installation

#### Docker Compose

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:v2.45.0
    container_name: carla-rl-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./monitoring/prometheus/rules:/etc/prometheus/rules:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
      - '--web.external-url=http://localhost:9090'
      - '--storage.tsdb.retention.time=30d'
      - '--storage.tsdb.retention.size=10GB'
    networks:
      - monitoring
    restart: unless-stopped

  node-exporter:
    image: prom/node-exporter:v1.6.1
    container_name: carla-rl-node-exporter
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    networks:
      - monitoring
    restart: unless-stopped

volumes:
  prometheus-data:
    driver: local

networks:
  monitoring:
    driver: bridge
```

#### Bare Metal Installation

```bash
# Download and install Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz
tar xvf prometheus-2.45.0.linux-amd64.tar.gz
sudo cp prometheus-2.45.0.linux-amd64/prometheus /usr/local/bin/
sudo cp prometheus-2.45.0.linux-amd64/promtool /usr/local/bin/

# Create Prometheus user
sudo useradd --no-create-home --shell /bin/false prometheus

# Create directories
sudo mkdir /etc/prometheus
sudo mkdir /var/lib/prometheus
sudo chown prometheus:prometheus /etc/prometheus
sudo chown prometheus:prometheus /var/lib/prometheus
```

### 2. Configuration

Create `monitoring/prometheus/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'carla-rl-cluster'
    environment: 'production'

rule_files:
  - "rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  # Prometheus itself
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # CarlaRL Policy-as-a-Service
  - job_name: 'carla-rl-serving'
    static_configs:
      - targets: ['carla-rl-serving:8080']
    metrics_path: '/metrics'
    scrape_interval: 5s
    scrape_timeout: 5s

  # Node Exporter
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
    scrape_interval: 15s

  # Additional services
  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:9113']
    scrape_interval: 15s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
    scrape_interval: 15s
```

### 3. Alert Rules

Create `monitoring/prometheus/rules/carla_rl_alerts.yml`:

```yaml
groups:
  - name: carla-rl-serving
    rules:
      # High latency alert
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.01
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
          description: "P95 latency is above 10ms for 5 minutes"

      # High error rate alert
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is above 5% for 5 minutes"

      # Service down alert
      - alert: ServiceDown
        expr: up{job="carla-rl-serving"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "CarlaRL service is down"
          description: "CarlaRL service has been down for more than 1 minute"

      # High memory usage alert
      - alert: HighMemoryUsage
        expr: process_resident_memory_bytes / 1024 / 1024 > 1000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is above 1GB for 5 minutes"

      # High CPU usage alert
      - alert: HighCPUUsage
        expr: rate(process_cpu_seconds_total[5m]) * 100 > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage"
          description: "CPU usage is above 80% for 5 minutes"

      # Model loading failure alert
      - alert: ModelLoadingFailure
        expr: increase(model_loading_errors_total[5m]) > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Model loading failure"
          description: "Model loading has failed in the last 5 minutes"
```

## Grafana Setup

### 1. Installation

#### Docker Compose

```yaml
# Add to docker-compose.monitoring.yml
  grafana:
    image: grafana/grafana:10.1.0
    container_name: carla-rl-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_INSTALL_PLUGINS=grafana-piechart-panel
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards:ro
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
    networks:
      - monitoring
    restart: unless-stopped
    depends_on:
      - prometheus
```

#### Bare Metal Installation

```bash
# Install Grafana
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
echo "deb https://packages.grafana.com/oss/deb stable main" | sudo tee -a /etc/apt/sources.list.d/grafana.list
sudo apt update
sudo apt install -y grafana

# Start Grafana
sudo systemctl enable grafana-server
sudo systemctl start grafana-server
```

### 2. Data Source Configuration

Create `monitoring/grafana/provisioning/datasources/prometheus.yml`:

```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
    jsonData:
      timeInterval: "5s"
      queryTimeout: "60s"
      httpMethod: "POST"
```

### 3. Dashboard Configuration

Create `monitoring/grafana/provisioning/dashboards/dashboards.yml`:

```yaml
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
```

### 4. CarlaRL Dashboard

Create `monitoring/grafana/dashboards/carla-rl-dashboard.json`:

```json
{
  "dashboard": {
    "id": null,
    "title": "CarlaRL Policy-as-a-Service",
    "tags": ["carla-rl", "ml-serving"],
    "style": "dark",
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}",
            "refId": "A"
          }
        ],
        "xAxis": {
          "show": true,
          "mode": "time"
        },
        "yAxes": [
          {
            "label": "Requests/sec",
            "show": true
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 0
        }
      },
      {
        "id": 2,
        "title": "Latency Percentiles",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P50",
            "refId": "A"
          },
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P95",
            "refId": "B"
          },
          {
            "expr": "histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P99",
            "refId": "C"
          }
        ],
        "xAxis": {
          "show": true,
          "mode": "time"
        },
        "yAxes": [
          {
            "label": "Latency (seconds)",
            "show": true,
            "logBase": 1,
            "min": 0
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 0
        }
      },
      {
        "id": 3,
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m])",
            "legendFormat": "Error Rate",
            "refId": "A"
          }
        ],
        "xAxis": {
          "show": true,
          "mode": "time"
        },
        "yAxes": [
          {
            "label": "Error Rate",
            "show": true,
            "min": 0,
            "max": 1
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 8
        }
      },
      {
        "id": 4,
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "process_resident_memory_bytes / 1024 / 1024",
            "legendFormat": "Memory (MB)",
            "refId": "A"
          }
        ],
        "xAxis": {
          "show": true,
          "mode": "time"
        },
        "yAxes": [
          {
            "label": "Memory (MB)",
            "show": true
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 8
        }
      },
      {
        "id": 5,
        "title": "CPU Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(process_cpu_seconds_total[5m]) * 100",
            "legendFormat": "CPU %",
            "refId": "A"
          }
        ],
        "xAxis": {
          "show": true,
          "mode": "time"
        },
        "yAxes": [
          {
            "label": "CPU %",
            "show": true,
            "min": 0,
            "max": 100
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 16
        }
      },
      {
        "id": 6,
        "title": "Model Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(model_inference_duration_seconds[5m])",
            "legendFormat": "Inference Duration",
            "refId": "A"
          },
          {
            "expr": "rate(model_inference_requests_total[5m])",
            "legendFormat": "Inference Requests",
            "refId": "B"
          }
        ],
        "xAxis": {
          "show": true,
          "mode": "time"
        },
        "yAxes": [
          {
            "label": "Rate",
            "show": true
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 16
        }
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}
```

## Alerting Setup

### 1. AlertManager Configuration

Create `monitoring/alertmanager/alertmanager.yml`:

```yaml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@carla-rl.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
  - name: 'web.hook'
    webhook_configs:
      - url: 'http://localhost:5001/'

  - name: 'email'
    email_configs:
      - to: 'admin@carla-rl.com'
        subject: 'CarlaRL Alert: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}

  - name: 'slack'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        channel: '#alerts'
        title: 'CarlaRL Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
```

### 2. AlertManager Docker Service

```yaml
# Add to docker-compose.monitoring.yml
  alertmanager:
    image: prom/alertmanager:v0.25.0
    container_name: carla-rl-alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro
      - alertmanager-data:/alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
      - '--web.external-url=http://localhost:9093'
    networks:
      - monitoring
    restart: unless-stopped
```

## Logging Setup

### 1. Structured Logging

```python
# src/monitoring/logging.py
import logging
import json
import uuid
from datetime import datetime
from typing import Dict, Any

class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create handler
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log(self, level: str, message: str, **kwargs):
        """Log structured message."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level,
            'message': message,
            'correlation_id': kwargs.get('correlation_id', str(uuid.uuid4())),
            **kwargs
        }
        
        if level == 'ERROR':
            self.logger.error(json.dumps(log_entry))
        elif level == 'WARNING':
            self.logger.warning(json.dumps(log_entry))
        elif level == 'INFO':
            self.logger.info(json.dumps(log_entry))
        else:
            self.logger.debug(json.dumps(log_entry))
```

### 2. Log Aggregation (ELK Stack)

```yaml
# docker-compose.logging.yml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.8.0
    container_name: carla-rl-elasticsearch
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch-data:/usr/share/elasticsearch/data
    networks:
      - logging

  logstash:
    image: docker.elastic.co/logstash/logstash:8.8.0
    container_name: carla-rl-logstash
    ports:
      - "5044:5044"
    volumes:
      - ./monitoring/logstash/logstash.conf:/usr/share/logstash/pipeline/logstash.conf:ro
    networks:
      - logging
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:8.8.0
    container_name: carla-rl-kibana
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    networks:
      - logging
    depends_on:
      - elasticsearch

volumes:
  elasticsearch-data:
    driver: local

networks:
  logging:
    driver: bridge
```

## Tracing Setup

### 1. Jaeger Configuration

```yaml
# Add to docker-compose.monitoring.yml
  jaeger:
    image: jaegertracing/all-in-one:1.47
    container_name: carla-rl-jaeger
    ports:
      - "16686:16686"
      - "14268:14268"
    environment:
      - COLLECTOR_OTLP_ENABLED=true
    networks:
      - monitoring
    restart: unless-stopped
```

### 2. Application Tracing

```python
# src/monitoring/tracing.py
import opentelemetry
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

def setup_tracing(service_name: str, jaeger_endpoint: str):
    """Setup distributed tracing."""
    # Create tracer provider
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    
    # Create Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name="jaeger",
        agent_port=14268,
    )
    
    # Create span processor
    span_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)
    RequestsInstrumentor().instrument()
    
    return tracer
```

## Custom Metrics

### 1. Application Metrics

```python
# src/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Request metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
REQUEST_SIZE = Histogram('http_request_size_bytes', 'HTTP request size', ['method', 'endpoint'])
RESPONSE_SIZE = Histogram('http_response_size_bytes', 'HTTP response size', ['method', 'endpoint'])

# Model metrics
MODEL_INFERENCE_COUNT = Counter('model_inference_requests_total', 'Total model inference requests', ['model_version'])
MODEL_INFERENCE_DURATION = Histogram('model_inference_duration_seconds', 'Model inference duration', ['model_version'])
MODEL_LOAD_TIME = Histogram('model_load_duration_seconds', 'Model load duration', ['model_version'])
MODEL_MEMORY_USAGE = Gauge('model_memory_usage_bytes', 'Model memory usage', ['model_version'])

# System metrics
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Active connections')
QUEUE_SIZE = Gauge('queue_size', 'Queue size')
CACHE_HIT_RATIO = Gauge('cache_hit_ratio', 'Cache hit ratio')

def start_metrics_server(port: int = 9090):
    """Start Prometheus metrics server."""
    start_http_server(port)
```

### 2. Custom Metrics Collection

```python
# src/monitoring/custom_metrics.py
import psutil
import torch
from prometheus_client import Gauge

# System metrics
CPU_USAGE = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
MEMORY_USAGE = Gauge('system_memory_usage_bytes', 'Memory usage in bytes')
DISK_USAGE = Gauge('system_disk_usage_bytes', 'Disk usage in bytes')

# GPU metrics (if available)
if torch.cuda.is_available():
    GPU_MEMORY_USAGE = Gauge('gpu_memory_usage_bytes', 'GPU memory usage', ['device'])
    GPU_UTILIZATION = Gauge('gpu_utilization_percent', 'GPU utilization percentage', ['device'])

def collect_system_metrics():
    """Collect system metrics."""
    # CPU usage
    CPU_USAGE.set(psutil.cpu_percent())
    
    # Memory usage
    memory = psutil.virtual_memory()
    MEMORY_USAGE.set(memory.used)
    
    # Disk usage
    disk = psutil.disk_usage('/')
    DISK_USAGE.set(disk.used)
    
    # GPU metrics
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            GPU_MEMORY_USAGE.labels(device=i).set(torch.cuda.memory_allocated(i))
            GPU_UTILIZATION.labels(device=i).set(torch.cuda.utilization(i))
```

## Dashboards

### 1. System Overview Dashboard

```json
{
  "dashboard": {
    "title": "System Overview",
    "panels": [
      {
        "title": "System Load",
        "type": "graph",
        "targets": [
          {
            "expr": "node_load1",
            "legendFormat": "1m Load"
          },
          {
            "expr": "node_load5",
            "legendFormat": "5m Load"
          },
          {
            "expr": "node_load15",
            "legendFormat": "15m Load"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes",
            "legendFormat": "Used Memory"
          },
          {
            "expr": "node_memory_MemAvailable_bytes",
            "legendFormat": "Available Memory"
          }
        ]
      }
    ]
  }
}
```

### 2. Application Performance Dashboard

```json
{
  "dashboard": {
    "title": "Application Performance",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P95"
          },
          {
            "expr": "histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P99"
          }
        ]
      }
    ]
  }
}
```

## Troubleshooting

### 1. Prometheus Issues

```bash
# Check Prometheus status
curl http://localhost:9090/-/healthy

# Check targets
curl http://localhost:9090/api/v1/targets

# Check configuration
promtool check config /etc/prometheus/prometheus.yml
```

### 2. Grafana Issues

```bash
# Check Grafana status
curl http://localhost:3000/api/health

# Check data sources
curl http://localhost:3000/api/datasources

# Check dashboards
curl http://localhost:3000/api/dashboards
```

### 3. Alerting Issues

```bash
# Check AlertManager status
curl http://localhost:9093/-/healthy

# Check alerts
curl http://localhost:9093/api/v1/alerts

# Check silences
curl http://localhost:9093/api/v1/silences
```

## Best Practices

### 1. Metric Naming

- Use descriptive names: `http_requests_total` not `requests`
- Include units: `duration_seconds`, `size_bytes`
- Use consistent naming conventions
- Avoid high cardinality labels

### 2. Alerting

- Set appropriate thresholds
- Use multiple severity levels
- Include runbook links in alerts
- Test alerting regularly

### 3. Dashboard Design

- Keep dashboards focused and relevant
- Use appropriate visualization types
- Include time ranges and refresh intervals
- Organize panels logically

### 4. Logging

- Use structured logging (JSON)
- Include correlation IDs
- Set appropriate log levels
- Rotate logs regularly

## Next Steps

- [Performance Tuning Guide](performance-tuning/performance-tuning.md)
- [Configuration Reference](configuration-reference.md)
- [Troubleshooting Guide](troubleshooting/troubleshooting.md)
- [Deployment Guides](deployment-guides/)
