# Troubleshooting Guide

This guide helps diagnose and resolve common issues with the CarlaRL Policy-as-a-Service system.

## Table of Contents

- [Quick Diagnosis](#quick-diagnosis)
- [Common Issues](#common-issues)
- [Performance Issues](#performance-issues)
- [Deployment Issues](#deployment-issues)
- [Configuration Issues](#configuration-issues)
- [Monitoring and Debugging](#monitoring-and-debugging)
- [Recovery Procedures](#recovery-procedures)
- [Getting Help](#getting-help)

## Quick Diagnosis

### Health Check Commands

```bash
# Check service status
curl http://localhost:8080/healthz

# Check service metadata
curl http://localhost:8080/metadata

# Check metrics
curl http://localhost:8080/metrics

# Test prediction endpoint
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"observations": [{"speed": 25.5, "steering": 0.1, "sensors": [0.8, 0.2, 0.5, 0.9, 0.1]}], "deterministic": true}'
```

### Log Analysis

```bash
# View recent logs
tail -f logs/app.log

# Search for errors
grep -i error logs/app.log

# Search for warnings
grep -i warning logs/app.log

# View system logs
journalctl -u carla-rl-serving -f
```

### Resource Monitoring

```bash
# Check CPU and memory usage
htop

# Check disk usage
df -h

# Check network connections
netstat -tulpn | grep 8080

# Check GPU usage (if applicable)
nvidia-smi
```

## Common Issues

### 1. Service Won't Start

#### Symptoms
- Service fails to start
- Error messages in logs
- Port already in use

#### Diagnosis
```bash
# Check if port is in use
sudo netstat -tulpn | grep 8080
sudo lsof -i :8080

# Check service status
sudo systemctl status carla-rl-serving

# Check logs
sudo journalctl -u carla-rl-serving -n 50
```

#### Solutions

**Port Already in Use:**
```bash
# Kill process using port 8080
sudo fuser -k 8080/tcp

# Or find and kill specific process
sudo lsof -ti:8080 | xargs sudo kill -9
```

**Permission Issues:**
```bash
# Fix ownership
sudo chown -R carla-rl:carla-rl /home/carla-rl/carla-rl-serving

# Fix permissions
sudo chmod -R 755 /home/carla-rl/carla-rl-serving
```

**Missing Dependencies:**
```bash
# Install missing packages
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# Reinstall Python dependencies
cd /home/carla-rl/carla-rl-serving
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Model Loading Errors

#### Symptoms
- "Model not found" errors
- "Invalid model format" errors
- "CUDA out of memory" errors

#### Diagnosis
```bash
# Check if model files exist
ls -la artifacts/v0.1.0/

# Check model file integrity
python -c "import torch; print(torch.jit.load('artifacts/v0.1.0/model.pt'))"

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

#### Solutions

**Missing Model Files:**
```bash
# Create example artifacts
python scripts/create_example_artifacts.py

# Or download from repository
wget https://your-repo.com/models/v0.1.0/model.pt -O artifacts/v0.1.0/model.pt
wget https://your-repo.com/models/v0.1.0/preprocessor.pkl -O artifacts/v0.1.0/preprocessor.pkl
```

**Invalid Model Format:**
```bash
# Check model format
python -c "
import torch
model = torch.jit.load('artifacts/v0.1.0/model.pt')
print('Model loaded successfully')
print('Model type:', type(model))
"

# Convert model if needed
python scripts/convert_model.py --input model.pth --output model.pt
```

**CUDA Out of Memory:**
```bash
# Reduce batch size
export MODEL_BATCH_SIZE=1

# Use CPU instead
export MODEL_DEVICE=cpu

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"
```

### 3. High Latency

#### Symptoms
- Response times > 10ms
- Timeout errors
- Slow predictions

#### Diagnosis
```bash
# Run benchmark
python scripts/run_benchmarks.py

# Check system resources
htop
iostat -x 1

# Check network latency
ping localhost
```

#### Solutions

**CPU Optimization:**
```bash
# Increase workers
export SERVER_WORKERS=4

# Enable CPU optimizations
export MODEL_OPTIMIZE=true
export MODEL_USE_AVX=true
export MODEL_USE_MKL=true
```

**Memory Optimization:**
```bash
# Enable memory pinning
export MODEL_MEMORY_PINNING=true

# Increase cache size
export MODEL_CACHE_SIZE=1000

# Use smaller batch size
export MODEL_BATCH_SIZE=1
```

**Network Optimization:**
```bash
# Use localhost for testing
curl http://127.0.0.1:8080/healthz

# Check network configuration
sudo ethtool eth0
```

### 4. Memory Issues

#### Symptoms
- Out of memory errors
- High memory usage
- Memory leaks

#### Diagnosis
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head

# Check for memory leaks
python scripts/memory_profiler.py

# Monitor memory over time
watch -n 1 'free -h'
```

#### Solutions

**Reduce Memory Usage:**
```bash
# Use smaller batch size
export MODEL_BATCH_SIZE=1

# Disable model caching
export MODEL_CACHE_MODELS=false

# Use CPU instead of GPU
export MODEL_DEVICE=cpu
```

**Fix Memory Leaks:**
```bash
# Restart service
sudo systemctl restart carla-rl-serving

# Clear caches
python -c "import gc; gc.collect()"

# Monitor memory usage
python scripts/memory_monitor.py
```

### 5. Authentication Errors

#### Symptoms
- 401 Unauthorized errors
- API key validation failures
- CORS errors

#### Diagnosis
```bash
# Check API key
curl -H "X-API-Key: your-api-key" http://localhost:8080/healthz

# Check CORS headers
curl -H "Origin: https://yourdomain.com" -v http://localhost:8080/healthz
```

#### Solutions

**Invalid API Key:**
```bash
# Check configuration
grep -i api_key config/production.yaml

# Update API key
export SECURITY_API_KEYS='["new-api-key"]'

# Restart service
sudo systemctl restart carla-rl-serving
```

**CORS Issues:**
```bash
# Update CORS origins
export SECURITY_CORS_ORIGINS='["https://yourdomain.com"]'

# Restart service
sudo systemctl restart carla-rl-serving
```

## Performance Issues

### 1. Low Throughput

#### Symptoms
- < 1000 requests/second
- Queue buildup
- High response times

#### Diagnosis
```bash
# Run throughput test
python scripts/run_benchmarks.py --test-type throughput

# Check worker count
ps aux | grep uvicorn

# Monitor system resources
htop
iostat -x 1
```

#### Solutions

**Increase Workers:**
```bash
# Set appropriate worker count
export SERVER_WORKERS=4

# Restart service
sudo systemctl restart carla-rl-serving
```

**Optimize Batch Processing:**
```bash
# Increase batch size
export MODEL_BATCH_SIZE=4

# Enable batch optimization
export MODEL_OPTIMIZE=true
```

**Network Optimization:**
```bash
# Use keep-alive connections
export SERVER_KEEP_ALIVE=true

# Increase connection limits
export SERVER_MAX_CONNECTIONS=2000
```

### 2. High CPU Usage

#### Symptoms
- CPU usage > 80%
- Slow response times
- System unresponsive

#### Diagnosis
```bash
# Check CPU usage
top -p $(pgrep -f uvicorn)

# Check for CPU-intensive operations
perf top

# Monitor system load
uptime
```

#### Solutions

**Reduce CPU Load:**
```bash
# Reduce workers
export SERVER_WORKERS=2

# Use CPU optimization
export MODEL_USE_AVX=true
export MODEL_USE_MKL=true
```

**Optimize Model:**
```bash
# Use optimized model format
export MODEL_BACKEND=torchscript

# Enable JIT compilation
export MODEL_JIT=true
```

### 3. High Memory Usage

#### Symptoms
- Memory usage > 80%
- Out of memory errors
- System swapping

#### Diagnosis
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head

# Check for memory leaks
python scripts/memory_profiler.py
```

#### Solutions

**Reduce Memory Usage:**
```bash
# Use smaller batch size
export MODEL_BATCH_SIZE=1

# Disable model caching
export MODEL_CACHE_MODELS=false

# Use CPU instead of GPU
export MODEL_DEVICE=cpu
```

**Memory Optimization:**
```bash
# Enable memory pinning
export MODEL_MEMORY_PINNING=true

# Set memory limits
export MODEL_MAX_MEMORY=2GB
```

## Deployment Issues

### 1. Docker Issues

#### Symptoms
- Container won't start
- Image build failures
- Port binding errors

#### Diagnosis
```bash
# Check container status
docker ps -a

# Check container logs
docker logs carla-rl-serving

# Check image
docker images | grep carla-rl
```

#### Solutions

**Container Won't Start:**
```bash
# Check Docker daemon
sudo systemctl status docker

# Restart Docker
sudo systemctl restart docker

# Rebuild image
docker build --no-cache -t carla-rl-serving .
```

**Port Binding Issues:**
```bash
# Check port usage
sudo netstat -tulpn | grep 8080

# Use different port
docker run -p 8081:8080 carla-rl-serving
```

### 2. Kubernetes Issues

#### Symptoms
- Pods not starting
- Service not accessible
- Image pull errors

#### Diagnosis
```bash
# Check pod status
kubectl get pods -n carla-rl-serving

# Check pod logs
kubectl logs -f deployment/model-serving -n carla-rl-serving

# Check service
kubectl get svc -n carla-rl-serving
```

#### Solutions

**Pod Not Starting:**
```bash
# Check pod events
kubectl describe pod <pod-name> -n carla-rl-serving

# Check resource limits
kubectl top pods -n carla-rl-serving

# Update resource limits
kubectl patch deployment model-serving -n carla-rl-serving -p '{"spec":{"template":{"spec":{"containers":[{"name":"model-serving","resources":{"limits":{"memory":"2Gi"}}}]}}}}'
```

**Image Pull Errors:**
```bash
# Check image pull secrets
kubectl get secrets -n carla-rl-serving

# Update image
kubectl set image deployment/model-serving model-serving=your-registry/carla-rl-serving:latest -n carla-rl-serving
```

### 3. Bare Metal Issues

#### Symptoms
- Service won't start
- Permission errors
- Missing dependencies

#### Diagnosis
```bash
# Check service status
sudo systemctl status carla-rl-serving

# Check logs
sudo journalctl -u carla-rl-serving -n 50

# Check permissions
ls -la /home/carla-rl/carla-rl-serving
```

#### Solutions

**Permission Issues:**
```bash
# Fix ownership
sudo chown -R carla-rl:carla-rl /home/carla-rl/carla-rl-serving

# Fix permissions
sudo chmod -R 755 /home/carla-rl/carla-rl-serving
```

**Missing Dependencies:**
```bash
# Install system packages
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# Install Python packages
cd /home/carla-rl/carla-rl-serving
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration Issues

### 1. Invalid Configuration

#### Symptoms
- Configuration validation errors
- Service won't start
- Unexpected behavior

#### Diagnosis
```bash
# Validate configuration
python scripts/validate_config.py

# Check configuration file
python -c "from src.config.settings import AppConfig; print(AppConfig())"
```

#### Solutions

**Fix Configuration:**
```bash
# Check configuration file syntax
python -m yaml config/production.yaml

# Update configuration
nano config/production.yaml

# Restart service
sudo systemctl restart carla-rl-serving
```

### 2. Environment Variable Issues

#### Symptoms
- Environment variables not loaded
- Wrong values used
- Configuration conflicts

#### Diagnosis
```bash
# Check environment variables
env | grep -i carla

# Check .env file
cat .env

# Test configuration loading
python -c "from src.config.settings import AppConfig; print(AppConfig().model_dump())"
```

#### Solutions

**Fix Environment Variables:**
```bash
# Update .env file
nano .env

# Reload environment
source .env

# Restart service
sudo systemctl restart carla-rl-serving
```

## Monitoring and Debugging

### 1. Enable Debug Logging

```bash
# Set debug level
export LOGGING_LEVEL=DEBUG

# Enable debug mode
export DEBUG=true

# Restart service
sudo systemctl restart carla-rl-serving
```

### 2. Enable Profiling

```bash
# Enable CPU profiling
export ENABLE_PROFILING=true

# Enable memory profiling
export ENABLE_MEMORY_PROFILING=true

# Run with profiler
python -m cProfile -o profile.stats src/server.py
```

### 3. Monitor Metrics

```bash
# Check Prometheus metrics
curl http://localhost:9090/metrics

# Check application metrics
curl http://localhost:8080/metrics

# View Grafana dashboard
open http://localhost:3000
```

### 4. Trace Requests

```bash
# Enable request tracing
export ENABLE_TRACING=true

# Check Jaeger traces
open http://localhost:16686
```

## Recovery Procedures

### 1. Service Recovery

```bash
# Restart service
sudo systemctl restart carla-rl-serving

# Check status
sudo systemctl status carla-rl-serving

# View logs
sudo journalctl -u carla-rl-serving -f
```

### 2. Data Recovery

```bash
# Backup current data
cp -r data data.backup.$(date +%Y%m%d_%H%M%S)

# Restore from backup
cp -r data.backup.20240101_120000/* data/

# Restart service
sudo systemctl restart carla-rl-serving
```

### 3. Configuration Recovery

```bash
# Backup configuration
cp config/production.yaml config/production.yaml.backup

# Restore from backup
cp config/production.yaml.backup config/production.yaml

# Restart service
sudo systemctl restart carla-rl-serving
```

### 4. Complete Reset

```bash
# Stop service
sudo systemctl stop carla-rl-serving

# Remove data
rm -rf data/* logs/* temp/*

# Recreate artifacts
python scripts/create_example_artifacts.py

# Start service
sudo systemctl start carla-rl-serving
```

## Getting Help

### 1. Log Collection

```bash
# Collect logs
mkdir -p debug-logs
cp logs/app.log debug-logs/
sudo journalctl -u carla-rl-serving > debug-logs/systemd.log
ps aux > debug-logs/processes.txt
free -h > debug-logs/memory.txt
df -h > debug-logs/disk.txt
```

### 2. System Information

```bash
# Collect system info
uname -a > debug-logs/system.txt
lscpu > debug-logs/cpu.txt
lspci > debug-logs/hardware.txt
```

### 3. Configuration Export

```bash
# Export configuration
python -c "from src.config.settings import AppConfig; import json; print(json.dumps(AppConfig().model_dump(), indent=2))" > debug-logs/config.json
```

### 4. Contact Support

When contacting support, include:

1. **System Information**: OS, Python version, hardware specs
2. **Configuration**: Current configuration files
3. **Logs**: Application and system logs
4. **Error Messages**: Exact error messages and stack traces
5. **Steps to Reproduce**: Detailed steps to reproduce the issue
6. **Expected Behavior**: What should happen
7. **Actual Behavior**: What actually happens

### 5. Community Resources

- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check the complete documentation
- **Discord/Slack**: Join the community chat
- **Stack Overflow**: Search for similar issues

## Prevention

### 1. Regular Monitoring

```bash
# Set up monitoring alerts
# Monitor CPU, memory, disk usage
# Monitor response times and error rates
# Monitor log file sizes
```

### 2. Regular Backups

```bash
# Backup configuration
cp config/production.yaml config/backup-$(date +%Y%m%d).yaml

# Backup data
tar -czf data-backup-$(date +%Y%m%d).tar.gz data/

# Backup logs
tar -czf logs-backup-$(date +%Y%m%d).tar.gz logs/
```

### 3. Health Checks

```bash
# Set up health check script
#!/bin/bash
curl -f http://localhost:8080/healthz || exit 1
```

### 4. Performance Testing

```bash
# Regular performance tests
python scripts/run_benchmarks.py --schedule daily

# Load testing
python scripts/load_test.py --duration 300 --concurrent 100
```

## Next Steps

- [Performance Tuning Guide](performance-tuning/performance-tuning.md)
- [Monitoring Setup Guide](monitoring/monitoring-setup.md)
- [Configuration Reference](../configuration-reference.md)
- [Deployment Guides](../deployment-guides/)
