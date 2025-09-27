# CarlaRL Policy-as-a-Service Deployment

This directory contains deployment configurations for different environments.

## Directory Structure

```
deploy/
├── docker/
│   └── docker-compose.yml    # Full stack with monitoring
└── k8s/
    └── deployment.yaml       # Kubernetes manifests
```

## Quick Testing on OrbStack

### Option 1: Simple Docker Test (Recommended)
```bash
# Build, run, and test with one command
make test-docker
```

### Option 2: Docker Compose (Full Stack)
```bash
# Start with monitoring stack
make docker-compose-up

# Run validation tests
python3 scripts/cluster_validation.py --url http://localhost:8080

# Stop services
make docker-compose-down
```

### Option 3: Kubernetes Test
```bash
# Deploy to k8s and test
make test-k8s

# Clean up
make undeploy-k8s
```

## What Gets Tested

The `scripts/cluster_validation.py` runs comprehensive tests:

- ✅ Service health and availability
- ✅ Model metadata and configuration
- ✅ Single and batch predictions
- ✅ Performance and latency
- ✅ Error handling
- ✅ Concurrent request handling
- ✅ Metrics endpoint

## Results

Tests output a summary and optionally save detailed results to JSON for analysis.

**That's it!** The Makefile handles all the complexity. 🎯
