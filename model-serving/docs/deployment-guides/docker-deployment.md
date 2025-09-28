# Docker Deployment Guide

This guide covers deploying the CarlaRL Policy-as-a-Service using Docker and Docker Compose.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Production Deployment](#production-deployment)
- [Configuration](#configuration)
- [Monitoring Setup](#monitoring-setup)
- [Troubleshooting](#troubleshooting)
- [Performance Tuning](#performance-tuning)

## Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- At least 2GB RAM available
- 1GB disk space for images and volumes
- Network access for pulling base images

## Quick Start

### 1. Clone and Build

```bash
git clone <repository-url>
cd carla-rl-serving
make docker-build
```

### 2. Create Example Artifacts

```bash
make create-artifacts
```

### 3. Run Service

```bash
# Simple deployment
make docker-run

# Or with Docker Compose
docker-compose up -d
```

### 4. Verify Deployment

```bash
# Health check
curl http://localhost:8080/healthz

# Test prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "observations": [{
      "speed": 25.5,
      "steering": 0.1,
      "sensors": [0.8, 0.2, 0.5, 0.9, 0.1]
    }],
    "deterministic": true
  }'
```

## Production Deployment

### 1. Environment Configuration

Create a `.env` file for production settings:

```bash
# Production environment variables
ARTIFACT_DIR=/app/artifacts
MODEL_VERSION=v0.1.0
USE_GPU=0
LOG_LEVEL=info
WORKERS=4
CORS_ORIGINS=https://yourdomain.com
ALLOWED_HOSTS=yourdomain.com,api.yourdomain.com
```

### 2. Production Docker Compose

```yaml
version: '3.8'

services:
  carla-rl-serving:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    container_name: carla-rl-serving-prod
    ports:
      - "8080:8080"
    environment:
      - ARTIFACT_DIR=/app/artifacts
      - MODEL_VERSION=${MODEL_VERSION}
      - USE_GPU=${USE_GPU}
      - LOG_LEVEL=${LOG_LEVEL}
      - WORKERS=${WORKERS}
      - CORS_ORIGINS=${CORS_ORIGINS}
      - ALLOWED_HOSTS=${ALLOWED_HOSTS}
    volumes:
      - ./artifacts:/app/artifacts:ro
      - ./logs:/app/logs
      - ./config:/app/config:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    restart: unless-stopped
    networks:
      - carla-rl-network
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2.0'
        reservations:
          memory: 1G
          cpus: '1.0'

  # Load balancer
  nginx:
    image: nginx:alpine
    container_name: carla-rl-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - carla-rl-serving
    networks:
      - carla-rl-network

networks:
  carla-rl-network:
    driver: bridge
```

### 3. Nginx Configuration

Create `nginx/nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream carla_rl_backend {
        server carla-rl-serving:8080;
    }

    server {
        listen 80;
        server_name yourdomain.com;

        location / {
            proxy_pass http://carla_rl_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 5s;
            proxy_send_timeout 10s;
            proxy_read_timeout 10s;
        }

        # Health check endpoint
        location /healthz {
            proxy_pass http://carla_rl_backend/healthz;
            access_log off;
        }
    }
}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARTIFACT_DIR` | `/app/artifacts` | Directory containing model artifacts |
| `MODEL_VERSION` | `v0.1.0` | Model version to load |
| `USE_GPU` | `0` | Enable GPU inference (1 for enabled) |
| `PORT` | `8080` | Server port |
| `WORKERS` | `1` | Number of uvicorn workers |
| `LOG_LEVEL` | `info` | Logging level (debug, info, warning, error) |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |
| `ALLOWED_HOSTS` | `*` | Allowed host headers |

### Volume Mounts

- `./artifacts:/app/artifacts:ro` - Model artifacts (read-only)
- `./logs:/app/logs` - Application logs
- `./config:/app/config:ro` - Configuration files (read-only)

### Resource Limits

```yaml
deploy:
  resources:
    limits:
      memory: 2G      # Maximum memory usage
      cpus: '2.0'     # Maximum CPU usage
    reservations:
      memory: 1G      # Guaranteed memory
      cpus: '1.0'     # Guaranteed CPU
```

## Monitoring Setup

### 1. Full Monitoring Stack

```bash
# Start with monitoring stack
docker-compose --profile monitoring up -d
```

This includes:
- **Prometheus**: Metrics collection
- **Grafana**: Metrics visualization
- **Redis**: Caching (optional)
- **Traefik**: Load balancing (optional)

### 2. Access Monitoring

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Traefik Dashboard**: http://localhost:8081

### 3. Custom Monitoring

```yaml
# Add to docker-compose.yml
services:
  prometheus:
    image: prom/prometheus:v2.45.0
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.enable-lifecycle'

  grafana:
    image: grafana/grafana:10.1.0
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning:ro
```

## Troubleshooting

### Common Issues

#### 1. Container Won't Start

```bash
# Check container logs
docker logs carla-rl-serving

# Check if port is already in use
netstat -tulpn | grep 8080

# Check Docker daemon
docker system info
```

#### 2. Model Loading Errors

```bash
# Verify artifacts exist
docker exec carla-rl-serving ls -la /app/artifacts/

# Check artifact integrity
docker exec carla-rl-serving python -c "
import torch
print(torch.jit.load('/app/artifacts/v0.1.0/model.pt'))
"
```

#### 3. Performance Issues

```bash
# Check resource usage
docker stats carla-rl-serving

# Run benchmark inside container
docker exec carla-rl-serving python scripts/run_benchmarks.py

# Check memory usage
docker exec carla-rl-serving free -h
```

#### 4. Network Issues

```bash
# Test connectivity
docker exec carla-rl-serving curl -f http://localhost:8080/healthz

# Check network configuration
docker network ls
docker network inspect carla-rl-network
```

### Debug Mode

```bash
# Run with debug logging
docker run -e LOG_LEVEL=debug carla-rl-serving

# Interactive debugging
docker run -it --entrypoint /bin/bash carla-rl-serving
```

## Performance Tuning

### 1. CPU Optimization

```yaml
# Increase workers for CPU-bound workloads
environment:
  - WORKERS=4

# Use CPU-optimized base image
FROM python:3.11-slim-bullseye
```

### 2. Memory Optimization

```yaml
# Set memory limits
deploy:
  resources:
    limits:
      memory: 4G
    reservations:
      memory: 2G

# Enable memory optimization
environment:
  - ENABLE_MEMORY_PINNING=1
  - CACHE_SIZE=1000
```

### 3. GPU Support

```yaml
# Enable GPU support
environment:
  - USE_GPU=1

# Use GPU-enabled base image
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# Runtime configuration
runtime: nvidia
```

### 4. Network Optimization

```yaml
# Use host networking for better performance
network_mode: host

# Or configure custom network
networks:
  carla-rl-network:
    driver: bridge
    driver_opts:
      com.docker.network.bridge.enable_icc: "true"
      com.docker.network.bridge.enable_ip_masquerade: "true"
```

## Security Considerations

### 1. Non-root User

The container runs as a non-root user (`modelserving`) for security.

### 2. Read-only Filesystem

```yaml
# Mount artifacts as read-only
volumes:
  - ./artifacts:/app/artifacts:ro
```

### 3. Resource Limits

```yaml
# Prevent resource exhaustion
deploy:
  resources:
    limits:
      memory: 2G
      cpus: '2.0'
```

### 4. Network Security

```yaml
# Use internal networks
networks:
  carla-rl-network:
    internal: true
```

## Scaling

### Horizontal Scaling

```bash
# Scale to multiple instances
docker-compose up -d --scale carla-rl-serving=3

# Use load balancer
docker-compose --profile with-lb up -d
```

### Vertical Scaling

```yaml
# Increase resources
deploy:
  resources:
    limits:
      memory: 8G
      cpus: '4.0'
    reservations:
      memory: 4G
      cpus: '2.0'
```

## Backup and Recovery

### 1. Backup Artifacts

```bash
# Backup model artifacts
docker run --rm -v carla-rl-serving_artifacts:/data -v $(pwd):/backup alpine tar czf /backup/artifacts-backup.tar.gz -C /data .
```

### 2. Backup Configuration

```bash
# Backup configuration
docker run --rm -v carla-rl-serving_config:/data -v $(pwd):/backup alpine tar czf /backup/config-backup.tar.gz -C /data .
```

### 3. Restore

```bash
# Restore artifacts
docker run --rm -v carla-rl-serving_artifacts:/data -v $(pwd):/backup alpine tar xzf /backup/artifacts-backup.tar.gz -C /data

# Restore configuration
docker run --rm -v carla-rl-serving_config:/data -v $(pwd):/backup alpine tar xzf /backup/config-backup.tar.gz -C /data
```

## Maintenance

### 1. Updates

```bash
# Update service
docker-compose pull
docker-compose up -d

# Update with rebuild
docker-compose build --no-cache
docker-compose up -d
```

### 2. Log Rotation

```yaml
# Configure log rotation
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

### 3. Health Checks

```yaml
# Configure health checks
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8080/healthz"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

## Next Steps

- [Kubernetes Deployment Guide](kubernetes-deployment.md)
- [Bare Metal Deployment Guide](bare-metal-deployment.md)
- [Configuration Reference](../configuration-reference.md)
- [Performance Tuning Guide](../performance-tuning/performance-tuning.md)
