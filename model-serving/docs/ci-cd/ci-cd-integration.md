# CI/CD Integration Documentation

This guide covers integrating the CarlaRL Policy-as-a-Service with CI/CD pipelines for automated testing, performance validation, and deployment.

## Table of Contents

- [Overview](#overview)
- [GitHub Actions Setup](#github-actions-setup)
- [Performance Validation Pipeline](#performance-validation-pipeline)
- [Deployment Pipeline](#deployment-pipeline)
- [Quality Gates](#quality-gates)
- [Monitoring Integration](#monitoring-integration)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

The CI/CD pipeline provides:

- **Automated Testing**: Unit, integration, and performance tests
- **Performance Validation**: Automated benchmarking and regression detection
- **Quality Gates**: Enforce performance and quality standards
- **Automated Deployment**: Deploy to staging and production environments
- **Monitoring Integration**: Set up monitoring and alerting
- **Rollback Capabilities**: Automatic rollback on failures

### Pipeline Stages

1. **Build**: Compile, test, and package application
2. **Test**: Run unit and integration tests
3. **Performance**: Run performance benchmarks and validation
4. **Security**: Security scanning and vulnerability assessment
5. **Deploy**: Deploy to staging and production
6. **Monitor**: Set up monitoring and alerting
7. **Validate**: Post-deployment validation

## GitHub Actions Setup

### 1. Workflow Structure

Create `.github/workflows/ci-cd.yml`:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]
  release:
    types: [published]

env:
  PYTHON_VERSION: '3.11'
  NODE_VERSION: '18'
  DOCKER_REGISTRY: 'ghcr.io'
  IMAGE_NAME: 'carla-rl-serving'

jobs:
  # Build and test job
  build-and-test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.11', '3.12']
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Run linting
        run: |
          flake8 src/ tests/
          black --check src/ tests/
          isort --check-only src/ tests/
      
      - name: Run type checking
        run: |
          mypy src/
      
      - name: Run unit tests
        run: |
          pytest tests/ --cov=src/ --cov-report=xml --cov-report=html
      
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-umbrella
          fail_ci_if_error: true

  # Performance validation job
  performance-validation:
    runs-on: ubuntu-latest
    needs: build-and-test
    if: github.event_name == 'pull_request' || github.ref == 'refs/heads/main'
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      
      - name: Create test artifacts
        run: |
          python scripts/create_example_artifacts.py
      
      - name: Run performance benchmarks
        run: |
          python scripts/run_benchmarks.py --config config/ci-benchmark.yaml --output results.json
      
      - name: Validate performance requirements
        run: |
          python scripts/validate_performance.py --results results.json --thresholds config/performance-thresholds.yaml
      
      - name: Upload performance results
        uses: actions/upload-artifact@v3
        with:
          name: performance-results
          path: results.json

  # Security scanning job
  security-scan:
    runs-on: ubuntu-latest
    needs: build-and-test
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          format: 'sarif'
          output: 'trivy-results.sarif'
      
      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'
      
      - name: Run Bandit security linter
        run: |
          pip install bandit
          bandit -r src/ -f json -o bandit-results.json
      
      - name: Upload Bandit results
        uses: actions/upload-artifact@v3
        with:
          name: security-results
          path: bandit-results.json

  # Build Docker image job
  build-docker:
    runs-on: ubuntu-latest
    needs: [build-and-test, performance-validation]
    if: github.event_name == 'push'
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.DOCKER_REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.DOCKER_REGISTRY }}/${{ github.repository }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=raw,value=latest,enable={{is_default_branch}}
      
      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # Deploy to staging job
  deploy-staging:
    runs-on: ubuntu-latest
    needs: [build-docker, security-scan]
    if: github.ref == 'refs/heads/develop'
    environment: staging
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Deploy to staging
        run: |
          echo "Deploying to staging environment"
          # Add your deployment commands here
      
      - name: Run smoke tests
        run: |
          python scripts/smoke_tests.py --environment staging
      
      - name: Set up monitoring
        run: |
          python scripts/setup_monitoring.py --environment staging

  # Deploy to production job
  deploy-production:
    runs-on: ubuntu-latest
    needs: [deploy-staging]
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Deploy to production
        run: |
          echo "Deploying to production environment"
          # Add your deployment commands here
      
      - name: Run health checks
        run: |
          python scripts/health_checks.py --environment production
      
      - name: Set up monitoring
        run: |
          python scripts/setup_monitoring.py --environment production
      
      - name: Notify deployment
        run: |
          python scripts/notify_deployment.py --environment production --status success
```

### 2. Performance Validation Workflow

Create `.github/workflows/performance-validation.yml`:

```yaml
name: Performance Validation

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:
    inputs:
      environment:
        description: 'Environment to test'
        required: true
        default: 'staging'
        type: choice
        options:
          - staging
          - production

jobs:
  performance-benchmark:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      
      - name: Create test artifacts
        run: |
          python scripts/create_example_artifacts.py
      
      - name: Run comprehensive benchmarks
        run: |
          python scripts/run_benchmarks.py \
            --config config/performance-benchmark.yaml \
            --output performance-results.json \
            --environment ${{ github.event.inputs.environment || 'staging' }}
      
      - name: Compare with baseline
        run: |
          python scripts/compare_performance.py \
            --current performance-results.json \
            --baseline baselines/latest.json \
            --threshold 10
      
      - name: Generate performance report
        run: |
          python scripts/generate_performance_report.py \
            --results performance-results.json \
            --output performance-report.html
      
      - name: Upload performance report
        uses: actions/upload-artifact@v3
        with:
          name: performance-report
          path: performance-report.html
      
      - name: Comment PR with results
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const results = JSON.parse(fs.readFileSync('performance-results.json', 'utf8'));
            
            const comment = `## Performance Validation Results
            
            ### Latency Metrics
            - **P50**: ${results.latency.p50}ms
            - **P95**: ${results.latency.p95}ms
            - **P99**: ${results.latency.p99}ms
            
            ### Throughput
            - **Requests/sec**: ${results.throughput.rps}
            
            ### Memory Usage
            - **Peak Memory**: ${results.memory.peak}MB
            - **Average Memory**: ${results.memory.average}MB
            
            ### Status
            ${results.overall_success ? '✅ **PASSED**' : '❌ **FAILED**'}
            `;
            
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comment
            });
```

## Performance Validation Pipeline

### 1. Benchmark Configuration

Create `config/ci-benchmark.yaml`:

```yaml
# CI/CD benchmark configuration
benchmark:
  warmup_iterations: 10
  measurement_iterations: 100
  batch_sizes: [1, 4, 8, 16]
  concurrent_requests: 4
  duration_seconds: 30

performance_thresholds:
  p50_latency_ms: 10.0
  p95_latency_ms: 20.0
  p99_latency_ms: 50.0
  throughput_rps: 1000.0
  memory_usage_mb: 1024.0
  error_rate_percent: 1.0

hardware_requirements:
  min_cpu_cores: 4
  min_memory_gb: 8
  min_disk_gb: 20

test_scenarios:
  - name: "single_request"
    description: "Single request latency test"
    batch_size: 1
    iterations: 1000
    
  - name: "batch_processing"
    description: "Batch processing test"
    batch_size: 16
    iterations: 100
    
  - name: "concurrent_load"
    description: "Concurrent load test"
    batch_size: 1
    concurrent_requests: 10
    duration_seconds: 60
    
  - name: "memory_stress"
    description: "Memory stress test"
    batch_size: 32
    iterations: 500
    memory_limit_mb: 2048
```

### 2. Performance Validation Script

Create `scripts/validate_performance.py`:

```python
#!/usr/bin/env python3
"""
Performance validation script for CI/CD pipeline.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

class PerformanceValidator:
    def __init__(self, results_file: str, thresholds_file: str):
        self.results = self.load_results(results_file)
        self.thresholds = self.load_thresholds(thresholds_file)
        self.validation_results = {}
    
    def load_results(self, file_path: str) -> Dict[str, Any]:
        """Load performance results from file."""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def load_thresholds(self, file_path: str) -> Dict[str, Any]:
        """Load performance thresholds from file."""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def validate_latency(self) -> bool:
        """Validate latency requirements."""
        thresholds = self.thresholds['performance_thresholds']
        results = self.results['latency']
        
        p50_pass = results['p50_ms'] <= thresholds['p50_latency_ms']
        p95_pass = results['p95_ms'] <= thresholds['p95_latency_ms']
        p99_pass = results['p99_ms'] <= thresholds['p99_latency_ms']
        
        self.validation_results['latency'] = {
            'p50': {'value': results['p50_ms'], 'threshold': thresholds['p50_latency_ms'], 'passed': p50_pass},
            'p95': {'value': results['p95_ms'], 'threshold': thresholds['p95_latency_ms'], 'passed': p95_pass},
            'p99': {'value': results['p99_ms'], 'threshold': thresholds['p99_latency_ms'], 'passed': p99_pass},
            'overall': p50_pass and p95_pass and p99_pass
        }
        
        return self.validation_results['latency']['overall']
    
    def validate_throughput(self) -> bool:
        """Validate throughput requirements."""
        thresholds = self.thresholds['performance_thresholds']
        results = self.results['throughput']
        
        throughput_pass = results['rps'] >= thresholds['throughput_rps']
        
        self.validation_results['throughput'] = {
            'value': results['rps'],
            'threshold': thresholds['throughput_rps'],
            'passed': throughput_pass
        }
        
        return throughput_pass
    
    def validate_memory(self) -> bool:
        """Validate memory usage requirements."""
        thresholds = self.thresholds['performance_thresholds']
        results = self.results['memory']
        
        memory_pass = results['peak_mb'] <= thresholds['memory_usage_mb']
        
        self.validation_results['memory'] = {
            'value': results['peak_mb'],
            'threshold': thresholds['memory_usage_mb'],
            'passed': memory_pass
        }
        
        return memory_pass
    
    def validate_error_rate(self) -> bool:
        """Validate error rate requirements."""
        thresholds = self.thresholds['performance_thresholds']
        results = self.results.get('error_rate', {'percent': 0.0})
        
        error_rate_pass = results['percent'] <= thresholds['error_rate_percent']
        
        self.validation_results['error_rate'] = {
            'value': results['percent'],
            'threshold': thresholds['error_rate_percent'],
            'passed': error_rate_pass
        }
        
        return error_rate_pass
    
    def validate_all(self) -> bool:
        """Validate all performance requirements."""
        latency_pass = self.validate_latency()
        throughput_pass = self.validate_throughput()
        memory_pass = self.validate_memory()
        error_rate_pass = self.validate_error_rate()
        
        overall_pass = latency_pass and throughput_pass and memory_pass and error_rate_pass
        
        self.validation_results['overall'] = {
            'passed': overall_pass,
            'latency': latency_pass,
            'throughput': throughput_pass,
            'memory': memory_pass,
            'error_rate': error_rate_pass
        }
        
        return overall_pass
    
    def generate_report(self) -> str:
        """Generate validation report."""
        report = []
        report.append("## Performance Validation Report")
        report.append("")
        
        # Overall status
        status = "✅ PASSED" if self.validation_results['overall']['passed'] else "❌ FAILED"
        report.append(f"**Overall Status**: {status}")
        report.append("")
        
        # Detailed results
        for category, results in self.validation_results.items():
            if category == 'overall':
                continue
                
            report.append(f"### {category.title()}")
            report.append("")
            
            if isinstance(results, dict) and 'overall' in results:
                # Latency has multiple metrics
                for metric, data in results.items():
                    if metric == 'overall':
                        continue
                    status = "✅" if data['passed'] else "❌"
                    report.append(f"- **{metric.upper()}**: {data['value']:.2f}ms (threshold: {data['threshold']:.2f}ms) {status}")
            else:
                # Single metric
                status = "✅" if results['passed'] else "❌"
                report.append(f"- **Value**: {results['value']:.2f} (threshold: {results['threshold']:.2f}) {status}")
            
            report.append("")
        
        return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description='Validate performance results')
    parser.add_argument('--results', required=True, help='Performance results file')
    parser.add_argument('--thresholds', required=True, help='Performance thresholds file')
    parser.add_argument('--output', help='Output report file')
    
    args = parser.parse_args()
    
    validator = PerformanceValidator(args.results, args.thresholds)
    passed = validator.validate_all()
    
    report = validator.generate_report()
    print(report)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
    
    sys.exit(0 if passed else 1)

if __name__ == '__main__':
    main()
```

## Deployment Pipeline

### 1. Staging Deployment

Create `scripts/deploy_staging.py`:

```python
#!/usr/bin/env python3
"""
Deploy to staging environment.
"""

import subprocess
import sys
import argparse
from pathlib import Path

def deploy_staging():
    """Deploy to staging environment."""
    print("Deploying to staging environment...")
    
    # Update Kubernetes manifests
    subprocess.run([
        'kubectl', 'apply', '-f', 'deploy/k8s/staging/'
    ], check=True)
    
    # Wait for deployment
    subprocess.run([
        'kubectl', 'rollout', 'status', 'deployment/model-serving', '-n', 'staging'
    ], check=True)
    
    # Run health checks
    subprocess.run([
        'python', 'scripts/health_checks.py', '--environment', 'staging'
    ], check=True)
    
    print("Staging deployment completed successfully!")

if __name__ == '__main__':
    deploy_staging()
```

### 2. Production Deployment

Create `scripts/deploy_production.py`:

```python
#!/usr/bin/env python3
"""
Deploy to production environment.
"""

import subprocess
import sys
import argparse
from pathlib import Path

def deploy_production():
    """Deploy to production environment."""
    print("Deploying to production environment...")
    
    # Backup current deployment
    subprocess.run([
        'kubectl', 'get', 'deployment', 'model-serving', '-n', 'production', '-o', 'yaml',
        '>', 'backup/deployment-backup.yaml'
    ], shell=True, check=True)
    
    # Update Kubernetes manifests
    subprocess.run([
        'kubectl', 'apply', '-f', 'deploy/k8s/production/'
    ], check=True)
    
    # Wait for deployment
    subprocess.run([
        'kubectl', 'rollout', 'status', 'deployment/model-serving', '-n', 'production'
    ], check=True)
    
    # Run health checks
    subprocess.run([
        'python', 'scripts/health_checks.py', '--environment', 'production'
    ], check=True)
    
    # Run smoke tests
    subprocess.run([
        'python', 'scripts/smoke_tests.py', '--environment', 'production'
    ], check=True)
    
    print("Production deployment completed successfully!")

if __name__ == '__main__':
    deploy_production()
```

## Quality Gates

### 1. Performance Gates

```yaml
# config/quality-gates.yaml
performance_gates:
  latency:
    p50_max_ms: 10.0
    p95_max_ms: 20.0
    p99_max_ms: 50.0
  
  throughput:
    min_rps: 1000.0
  
  memory:
    max_usage_mb: 1024.0
  
  error_rate:
    max_percent: 1.0

security_gates:
  vulnerabilities:
    max_critical: 0
    max_high: 0
    max_medium: 5
  
  code_quality:
    min_coverage: 80.0
    max_complexity: 10.0
    max_duplication: 5.0

deployment_gates:
  health_checks:
    max_failure_rate: 0.0
    max_response_time_ms: 5000.0
  
  smoke_tests:
    max_failure_rate: 0.0
    max_execution_time_s: 300.0
```

### 2. Quality Gate Validation

Create `scripts/validate_quality_gates.py`:

```python
#!/usr/bin/env python3
"""
Validate quality gates for deployment.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any

class QualityGateValidator:
    def __init__(self, gates_file: str):
        self.gates = self.load_gates(gates_file)
        self.results = {}
    
    def load_gates(self, file_path: str) -> Dict[str, Any]:
        """Load quality gates from file."""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def validate_performance_gates(self, results: Dict[str, Any]) -> bool:
        """Validate performance gates."""
        gates = self.gates['performance_gates']
        passed = True
        
        # Validate latency
        latency = results.get('latency', {})
        if latency.get('p50_ms', 0) > gates['latency']['p50_max_ms']:
            print(f"❌ P50 latency {latency['p50_ms']}ms exceeds threshold {gates['latency']['p50_max_ms']}ms")
            passed = False
        
        if latency.get('p95_ms', 0) > gates['latency']['p95_max_ms']:
            print(f"❌ P95 latency {latency['p95_ms']}ms exceeds threshold {gates['latency']['p95_max_ms']}ms")
            passed = False
        
        if latency.get('p99_ms', 0) > gates['latency']['p99_max_ms']:
            print(f"❌ P99 latency {latency['p99_ms']}ms exceeds threshold {gates['latency']['p99_max_ms']}ms")
            passed = False
        
        # Validate throughput
        throughput = results.get('throughput', {})
        if throughput.get('rps', 0) < gates['throughput']['min_rps']:
            print(f"❌ Throughput {throughput['rps']} RPS below threshold {gates['throughput']['min_rps']} RPS")
            passed = False
        
        # Validate memory
        memory = results.get('memory', {})
        if memory.get('peak_mb', 0) > gates['memory']['max_usage_mb']:
            print(f"❌ Memory usage {memory['peak_mb']}MB exceeds threshold {gates['memory']['max_usage_mb']}MB")
            passed = False
        
        # Validate error rate
        error_rate = results.get('error_rate', {})
        if error_rate.get('percent', 0) > gates['error_rate']['max_percent']:
            print(f"❌ Error rate {error_rate['percent']}% exceeds threshold {gates['error_rate']['max_percent']}%")
            passed = False
        
        return passed
    
    def validate_security_gates(self, results: Dict[str, Any]) -> bool:
        """Validate security gates."""
        gates = self.gates['security_gates']
        passed = True
        
        # Validate vulnerabilities
        vulnerabilities = results.get('vulnerabilities', {})
        if vulnerabilities.get('critical', 0) > gates['vulnerabilities']['max_critical']:
            print(f"❌ Critical vulnerabilities {vulnerabilities['critical']} exceed threshold {gates['vulnerabilities']['max_critical']}")
            passed = False
        
        if vulnerabilities.get('high', 0) > gates['vulnerabilities']['max_high']:
            print(f"❌ High vulnerabilities {vulnerabilities['high']} exceed threshold {gates['vulnerabilities']['max_high']}")
            passed = False
        
        if vulnerabilities.get('medium', 0) > gates['vulnerabilities']['max_medium']:
            print(f"❌ Medium vulnerabilities {vulnerabilities['medium']} exceed threshold {gates['vulnerabilities']['max_medium']}")
            passed = False
        
        # Validate code quality
        code_quality = results.get('code_quality', {})
        if code_quality.get('coverage', 0) < gates['code_quality']['min_coverage']:
            print(f"❌ Code coverage {code_quality['coverage']}% below threshold {gates['code_quality']['min_coverage']}%")
            passed = False
        
        if code_quality.get('complexity', 0) > gates['code_quality']['max_complexity']:
            print(f"❌ Code complexity {code_quality['complexity']} exceeds threshold {gates['code_quality']['max_complexity']}")
            passed = False
        
        if code_quality.get('duplication', 0) > gates['code_quality']['max_duplication']:
            print(f"❌ Code duplication {code_quality['duplication']}% exceeds threshold {gates['code_quality']['max_duplication']}%")
            passed = False
        
        return passed
    
    def validate_deployment_gates(self, results: Dict[str, Any]) -> bool:
        """Validate deployment gates."""
        gates = self.gates['deployment_gates']
        passed = True
        
        # Validate health checks
        health_checks = results.get('health_checks', {})
        if health_checks.get('failure_rate', 0) > gates['health_checks']['max_failure_rate']:
            print(f"❌ Health check failure rate {health_checks['failure_rate']}% exceeds threshold {gates['health_checks']['max_failure_rate']}%")
            passed = False
        
        if health_checks.get('response_time_ms', 0) > gates['health_checks']['max_response_time_ms']:
            print(f"❌ Health check response time {health_checks['response_time_ms']}ms exceeds threshold {gates['health_checks']['max_response_time_ms']}ms")
            passed = False
        
        # Validate smoke tests
        smoke_tests = results.get('smoke_tests', {})
        if smoke_tests.get('failure_rate', 0) > gates['smoke_tests']['max_failure_rate']:
            print(f"❌ Smoke test failure rate {smoke_tests['failure_rate']}% exceeds threshold {gates['smoke_tests']['max_failure_rate']}%")
            passed = False
        
        if smoke_tests.get('execution_time_s', 0) > gates['smoke_tests']['max_execution_time_s']:
            print(f"❌ Smoke test execution time {smoke_tests['execution_time_s']}s exceeds threshold {gates['smoke_tests']['max_execution_time_s']}s")
            passed = False
        
        return passed
    
    def validate_all(self, results: Dict[str, Any]) -> bool:
        """Validate all quality gates."""
        performance_pass = self.validate_performance_gates(results)
        security_pass = self.validate_security_gates(results)
        deployment_pass = self.validate_deployment_gates(results)
        
        overall_pass = performance_pass and security_pass and deployment_pass
        
        print(f"\nQuality Gate Results:")
        print(f"Performance: {'✅ PASSED' if performance_pass else '❌ FAILED'}")
        print(f"Security: {'✅ PASSED' if security_pass else '❌ FAILED'}")
        print(f"Deployment: {'✅ PASSED' if deployment_pass else '❌ FAILED'}")
        print(f"Overall: {'✅ PASSED' if overall_pass else '❌ FAILED'}")
        
        return overall_pass

def main():
    parser = argparse.ArgumentParser(description='Validate quality gates')
    parser.add_argument('--gates', required=True, help='Quality gates file')
    parser.add_argument('--results', required=True, help='Test results file')
    
    args = parser.parse_args()
    
    validator = QualityGateValidator(args.gates)
    
    with open(args.results, 'r') as f:
        results = json.load(f)
    
    passed = validator.validate_all(results)
    
    sys.exit(0 if passed else 1)

if __name__ == '__main__':
    main()
```

## Monitoring Integration

### 1. Post-Deployment Monitoring

Create `scripts/setup_monitoring.py`:

```python
#!/usr/bin/env python3
"""
Set up monitoring and alerting after deployment.
"""

import subprocess
import sys
import argparse
from pathlib import Path

def setup_monitoring(environment: str):
    """Set up monitoring for the specified environment."""
    print(f"Setting up monitoring for {environment} environment...")
    
    # Deploy Prometheus
    subprocess.run([
        'kubectl', 'apply', '-f', f'monitoring/prometheus/{environment}/'
    ], check=True)
    
    # Deploy Grafana
    subprocess.run([
        'kubectl', 'apply', '-f', f'monitoring/grafana/{environment}/'
    ], check=True)
    
    # Deploy AlertManager
    subprocess.run([
        'kubectl', 'apply', '-f', f'monitoring/alertmanager/{environment}/'
    ], check=True)
    
    # Wait for deployments
    subprocess.run([
        'kubectl', 'rollout', 'status', 'deployment/prometheus', '-n', 'monitoring'
    ], check=True)
    
    subprocess.run([
        'kubectl', 'rollout', 'status', 'deployment/grafana', '-n', 'monitoring'
    ], check=True)
    
    subprocess.run([
        'kubectl', 'rollout', 'status', 'deployment/alertmanager', '-n', 'monitoring'
    ], check=True)
    
    # Configure dashboards
    subprocess.run([
        'python', 'scripts/configure_dashboards.py', '--environment', environment
    ], check=True)
    
    # Configure alerts
    subprocess.run([
        'python', 'scripts/configure_alerts.py', '--environment', environment
    ], check=True)
    
    print(f"Monitoring setup completed for {environment} environment!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set up monitoring')
    parser.add_argument('--environment', required=True, help='Environment name')
    
    args = parser.parse_args()
    setup_monitoring(args.environment)
```

### 2. Health Checks

Create `scripts/health_checks.py`:

```python
#!/usr/bin/env python3
"""
Health checks for deployed service.
"""

import requests
import time
import sys
import argparse
from typing import Dict, Any

class HealthChecker:
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.results = {}
    
    def check_health_endpoint(self) -> bool:
        """Check health endpoint."""
        try:
            response = requests.get(f"{self.base_url}/healthz", timeout=self.timeout)
            success = response.status_code == 200
            self.results['health_endpoint'] = {
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'success': success
            }
            return success
        except Exception as e:
            self.results['health_endpoint'] = {
                'error': str(e),
                'success': False
            }
            return False
    
    def check_metrics_endpoint(self) -> bool:
        """Check metrics endpoint."""
        try:
            response = requests.get(f"{self.base_url}/metrics", timeout=self.timeout)
            success = response.status_code == 200
            self.results['metrics_endpoint'] = {
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'success': success
            }
            return success
        except Exception as e:
            self.results['metrics_endpoint'] = {
                'error': str(e),
                'success': False
            }
            return False
    
    def check_prediction_endpoint(self) -> bool:
        """Check prediction endpoint."""
        try:
            payload = {
                "observations": [{
                    "speed": 25.5,
                    "steering": 0.1,
                    "sensors": [0.8, 0.2, 0.5, 0.9, 0.1]
                }],
                "deterministic": True
            }
            response = requests.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=self.timeout
            )
            success = response.status_code == 200
            self.results['prediction_endpoint'] = {
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'success': success
            }
            return success
        except Exception as e:
            self.results['prediction_endpoint'] = {
                'error': str(e),
                'success': False
            }
            return False
    
    def run_all_checks(self) -> bool:
        """Run all health checks."""
        health_pass = self.check_health_endpoint()
        metrics_pass = self.check_metrics_endpoint()
        prediction_pass = self.check_prediction_endpoint()
        
        overall_pass = health_pass and metrics_pass and prediction_pass
        
        self.results['overall'] = {
            'success': overall_pass,
            'health_endpoint': health_pass,
            'metrics_endpoint': metrics_pass,
            'prediction_endpoint': prediction_pass
        }
        
        return overall_pass
    
    def print_results(self):
        """Print health check results."""
        print("Health Check Results:")
        print("=" * 50)
        
        for check, result in self.results.items():
            if check == 'overall':
                continue
                
            status = "✅ PASSED" if result['success'] else "❌ FAILED"
            print(f"{check}: {status}")
            
            if 'response_time' in result:
                print(f"  Response time: {result['response_time']:.3f}s")
            
            if 'error' in result:
                print(f"  Error: {result['error']}")
        
        print("=" * 50)
        overall_status = "✅ PASSED" if self.results['overall']['success'] else "❌ FAILED"
        print(f"Overall: {overall_status}")

def main():
    parser = argparse.ArgumentParser(description='Run health checks')
    parser.add_argument('--environment', required=True, help='Environment name')
    parser.add_argument('--timeout', type=int, default=30, help='Request timeout in seconds')
    
    args = parser.parse_args()
    
    # Determine base URL based on environment
    if args.environment == 'staging':
        base_url = 'http://staging.carla-rl.com'
    elif args.environment == 'production':
        base_url = 'http://api.carla-rl.com'
    else:
        base_url = f'http://localhost:8080'
    
    checker = HealthChecker(base_url, args.timeout)
    success = checker.run_all_checks()
    checker.print_results()
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
```

## Best Practices

### 1. Pipeline Design

- **Fast Feedback**: Keep pipeline stages fast for quick feedback
- **Parallel Execution**: Run independent stages in parallel
- **Caching**: Cache dependencies and build artifacts
- **Incremental Testing**: Only run tests for changed components

### 2. Performance Validation

- **Baseline Comparison**: Compare against established baselines
- **Regression Detection**: Detect performance regressions early
- **Hardware Consistency**: Use consistent hardware for benchmarks
- **Statistical Significance**: Ensure benchmark results are statistically significant

### 3. Deployment Strategy

- **Blue-Green Deployment**: Use blue-green deployment for zero downtime
- **Canary Deployment**: Gradually roll out changes to a subset of users
- **Rollback Plan**: Have a clear rollback plan for failures
- **Monitoring**: Monitor deployments closely

### 4. Quality Gates

- **Fail Fast**: Fail the pipeline early on quality issues
- **Clear Thresholds**: Set clear, measurable quality thresholds
- **Regular Review**: Regularly review and update quality gates
- **Documentation**: Document quality gate requirements

## Troubleshooting

### 1. Pipeline Failures

```bash
# Check pipeline status
gh run list --workflow=ci-cd.yml

# View pipeline logs
gh run view <run-id> --log

# Rerun failed jobs
gh run rerun <run-id>
```

### 2. Performance Issues

```bash
# Check benchmark results
cat performance-results.json | jq '.'

# Compare with baseline
python scripts/compare_performance.py --current performance-results.json --baseline baselines/latest.json

# Generate performance report
python scripts/generate_performance_report.py --results performance-results.json
```

### 3. Deployment Issues

```bash
# Check deployment status
kubectl get deployments -n production

# Check pod logs
kubectl logs -f deployment/model-serving -n production

# Check service status
kubectl get services -n production
```

## Next Steps

- [Monitoring Setup Guide](monitoring/monitoring-setup.md)
- [Performance Tuning Guide](performance-tuning/performance-tuning.md)
- [Configuration Reference](configuration-reference.md)
- [Troubleshooting Guide](troubleshooting/troubleshooting.md)
