# CarlaRL Deployment Testing Guide

This guide covers comprehensive testing for both Docker Compose and Kubernetes deployments of the CarlaRL Policy-as-a-Service infrastructure.

## Overview

The testing framework provides:

- **Docker Compose Testing**: Complete testing of containerized deployments
- **Kubernetes Testing**: Full testing of Kubernetes cluster deployments
- **Integration Testing**: API functionality and service validation
- **Load Testing**: Performance and scalability validation
- **Monitoring Testing**: Observability and metrics validation
- **Performance Testing**: Benchmarking and optimization validation

## Quick Start

### Prerequisites

```bash
# Install dependencies
make install-deps

# Check all dependencies
make check-deps
```

### Run All Tests

```bash
# Run comprehensive test suite
make test-all

# Or use the Python script directly
python3 scripts/run_deployment_tests.py
```

### Run Specific Test Types

```bash
# Docker Compose only
make test-docker

# Kubernetes only
make test-k8s

# Integration tests only
make test-integration

# Load tests only
make test-load

# Monitoring tests only
make test-monitoring

# Performance tests only
make test-performance
```

## Testing Framework

### 1. Docker Compose Testing

#### Basic Testing
```bash
# Run basic Docker Compose tests
python3 scripts/test_docker_compose.py

# Run with specific profiles
python3 scripts/test_docker_compose.py --profiles testing load-testing monitoring
```

#### Available Profiles
- `testing`: Basic integration tests
- `load-testing`: Load and performance tests
- `monitoring`: Monitoring and observability tests
- `performance`: Performance benchmarking tests

#### Development Environment
```bash
# Start development environment
make start-docker-dev

# Stop development environment
make stop-docker-dev
```

#### Manual Testing
```bash
# Start services only
python3 scripts/test_docker_compose.py --start-only

# Stop services only
python3 scripts/test_docker_compose.py --stop-only
```

### 2. Kubernetes Testing

#### Basic Testing
```bash
# Run complete Kubernetes tests
python3 scripts/test_kubernetes.py

# Deploy only
python3 scripts/test_kubernetes.py --deploy-only

# Cleanup only
python3 scripts/test_kubernetes.py --cleanup-only
```

#### Custom Namespace
```bash
# Test in custom namespace
python3 scripts/test_kubernetes.py --namespace carla-rl-test
```

#### Manual Deployment
```bash
# Deploy to Kubernetes
kubectl apply -f deploy/k8s/test-deployment.yaml

# Check deployment status
kubectl get deployments
kubectl get pods
kubectl get services

# Port forward for testing
kubectl port-forward service/model-serving-test-service 8080:80

# Clean up
kubectl delete -f deploy/k8s/test-deployment.yaml
```

### 3. Integration Testing

The integration testing framework validates:

#### API Endpoints
- `/healthz` - Health check endpoint
- `/metadata` - Model metadata endpoint
- `/warmup` - Model warmup endpoint
- `/predict` - Inference endpoint
- `/metrics` - Prometheus metrics endpoint
- `/versions` - Version information endpoint

#### Test Scenarios
- Service startup and readiness
- API functionality validation
- Error handling and edge cases
- Performance under load
- Monitoring and observability

#### Running Integration Tests
```bash
# Run integration tests directly
python3 -m pytest tests/integration/test_deployment.py -v

# Run with specific service URL
python3 tests/integration/test_deployment.py --url http://localhost:8080

# Run with output file
python3 tests/integration/test_deployment.py --output test-results.json
```

### 4. Load Testing

#### Load Test Scenarios
- Concurrent request handling
- Batch size performance
- Memory usage under load
- Response time distribution
- Error rate under stress

#### Running Load Tests
```bash
# Run load tests
python3 scripts/load_test.py --url http://localhost:8080 --duration 300

# Run with specific parameters
python3 scripts/load_test.py \
  --url http://localhost:8080 \
  --duration 600 \
  --concurrent 10 \
  --batch-sizes 1,4,8,16 \
  --output load-test-results.json
```

### 5. Monitoring Testing

#### Monitoring Validation
- Prometheus metrics collection
- Grafana dashboard functionality
- Alert rule validation
- Log aggregation and analysis
- Health check monitoring

#### Running Monitoring Tests
```bash
# Run monitoring tests
python3 scripts/monitoring_test.py \
  --service-url http://localhost:8080 \
  --prometheus-url http://localhost:9090 \
  --grafana-url http://localhost:3000

# Run with output file
python3 scripts/monitoring_test.py \
  --service-url http://localhost:8080 \
  --output monitoring-results.json
```

### 6. Performance Testing

#### Performance Benchmarks
- Inference latency (P50, P95, P99)
- Throughput (requests/second)
- Memory usage patterns
- CPU utilization
- GPU utilization (if available)

#### Running Performance Tests
```bash
# Run performance benchmark
python3 scripts/performance_benchmark.py \
  --url http://localhost:8080 \
  --duration 300 \
  --output benchmark-results.json

# Run with specific parameters
python3 scripts/performance_benchmark.py \
  --url http://localhost:8080 \
  --duration 600 \
  --batch-sizes 1,2,4,8,16,32 \
  --concurrent 5,10,20 \
  --output detailed-benchmark.json
```

## Test Configuration

### Environment Variables

```bash
# Service configuration
export SERVICE_URL=http://localhost:8080
export PROMETHEUS_URL=http://localhost:9090
export GRAFANA_URL=http://localhost:3000

# Test configuration
export TEST_TIMEOUT=30
export TEST_RETRIES=3
export LOAD_TEST_DURATION=300
export PERFORMANCE_TEST_DURATION=600
```

### Configuration Files

#### Docker Compose Configuration
- `deploy/docker/docker-compose.test.yml` - Test configuration
- `deploy/docker/prometheus.yml` - Prometheus configuration
- `deploy/docker/grafana-dashboard.json` - Grafana dashboard

#### Kubernetes Configuration
- `deploy/k8s/test-deployment.yaml` - Test deployment
- `deploy/k8s/deployment.yaml` - Production deployment

## Test Results

### Output Formats

All tests generate JSON output with comprehensive results:

```json
{
  "summary": {
    "overall_success": true,
    "total_duration": 120.5,
    "docker_compose_success": true,
    "kubernetes_success": true
  },
  "docker_compose": {
    "success": true,
    "duration": 60.2,
    "health_checks": {...},
    "integration_tests": {...},
    "load_tests": {...}
  },
  "kubernetes": {
    "success": true,
    "duration": 60.3,
    "cluster_connected": true,
    "application_deployed": true,
    "deployment_ready": true
  }
}
```

### Result Analysis

#### Success Criteria
- All health checks pass
- Integration tests complete successfully
- Performance meets requirements
- Monitoring functions correctly
- No critical errors

#### Performance Requirements
- P95 latency < 50ms
- Throughput > 100 requests/second
- Memory usage < 1GB
- CPU usage < 80%

## Troubleshooting

### Common Issues

#### Docker Compose Issues
```bash
# Check service status
docker-compose -f deploy/docker/docker-compose.test.yml ps

# View logs
docker-compose -f deploy/docker/docker-compose.test.yml logs

# Restart services
docker-compose -f deploy/docker/docker-compose.test.yml restart
```

#### Kubernetes Issues
```bash
# Check pod status
kubectl get pods

# View pod logs
kubectl logs -l app=model-serving-test

# Check service status
kubectl get services

# Check deployment status
kubectl get deployments
```

#### Test Failures
```bash
# Run with verbose output
python3 scripts/test_docker_compose.py --verbose

# Run specific test
python3 -m pytest tests/integration/test_deployment.py::TestDeploymentTester::test_health_endpoint -v

# Check test logs
tail -f /tmp/test-results.log
```

### Debug Commands

```bash
# Check service health
curl http://localhost:8080/healthz

# Check metrics
curl http://localhost:8080/metrics

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Check Grafana health
curl http://localhost:3000/api/health
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Deployment Tests

on:
  push:
    branches: [main, dev]
  pull_request:
    branches: [main]

jobs:
  test-docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Docker Compose tests
        run: make test-docker

  test-kubernetes:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Kubernetes
        run: |
          # Setup minikube or kind
          minikube start
      - name: Run Kubernetes tests
        run: make test-k8s
```

### Jenkins Pipeline

```groovy
pipeline {
    agent any
    
    stages {
        stage('Test Docker Compose') {
            steps {
                sh 'make test-docker'
            }
        }
        
        stage('Test Kubernetes') {
            steps {
                sh 'make test-k8s'
            }
        }
        
        stage('Generate Report') {
            steps {
                sh 'python3 scripts/run_deployment_tests.py --output test-report.json'
                publishTestResults testResultsPattern: 'test-report.json'
            }
        }
    }
}
```

## Best Practices

### Testing Strategy
1. Run tests in isolated environments
2. Use consistent test data and configurations
3. Validate both happy path and error scenarios
4. Test with realistic load patterns
5. Monitor resource usage during tests

### Performance Testing
1. Establish baseline performance metrics
2. Test with various batch sizes and concurrency levels
3. Monitor memory and CPU usage patterns
4. Validate performance under sustained load
5. Test performance degradation scenarios

### Monitoring Testing
1. Verify all metrics are collected correctly
2. Test alerting rules and thresholds
3. Validate dashboard functionality
4. Test log aggregation and analysis
5. Verify health check accuracy

### Error Testing
1. Test with invalid inputs
2. Test service unavailability scenarios
3. Test resource exhaustion conditions
4. Test network connectivity issues
5. Test configuration errors

## Advanced Usage

### Custom Test Scenarios

```python
# Custom integration test
from tests.integration.test_deployment import DeploymentTester

tester = DeploymentTester("http://localhost:8080")
results = tester.run_all_tests()

# Custom load test
from scripts.load_test import LoadTester

load_tester = LoadTester("http://localhost:8080")
results = load_tester.run_load_test(
    duration=600,
    concurrent_users=20,
    batch_sizes=[1, 4, 8, 16]
)
```

### Custom Monitoring Tests

```python
# Custom monitoring validation
from scripts.monitoring_test import MonitoringTester

monitor_tester = MonitoringTester(
    service_url="http://localhost:8080",
    prometheus_url="http://localhost:9090"
)
results = monitor_tester.validate_metrics()
```

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review test logs and output files
3. Verify service configurations
4. Check dependency versions
5. Contact the development team

## Changelog

### v1.0.0
- Initial testing framework implementation
- Docker Compose testing support
- Kubernetes testing support
- Integration testing framework
- Load testing capabilities
- Monitoring validation
- Performance benchmarking
- Comprehensive documentation
