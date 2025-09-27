# CarlaRL Policy-as-a-Service

High-performance serving infrastructure for CarlaRL reinforcement learning policies. This service transforms trained RL models into production-ready microservices with millisecond-latency inference capabilities.

## 🚀 Features

- **High-Performance Inference**: Sub-10ms P50 latency for single predictions on CPU
- **Batch Processing**: Efficient batch inference supporting 1000+ requests/sec
- **Production Ready**: Enterprise-grade serving with health checks, metrics, and observability
- **Model Management**: Standardized artifact format with semantic versioning
- **Memory Optimization**: Tensor pre-allocation and memory pinning for reduced latency
- **Caching**: Intelligent result caching for identical inputs
- **Deterministic Mode**: Reproducible inference for research and testing
- **Multi-format Support**: PyTorch and TorchScript model formats
- **Container Native**: Docker-first deployment with Kubernetes support

## 📋 Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional)
- CUDA-capable GPU (optional, for GPU inference)

### Local Development

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd carla-rl-serving
   make install-dev
   ```

2. **Create Example Artifacts**
   ```bash
   make create-artifacts
   ```

3. **Run Development Server**
   ```bash
   make dev
   ```

4. **Test the API**
   ```bash
   # Health check
   curl http://localhost:8080/healthz
   
   # Model metadata
   curl http://localhost:8080/metadata
   
   # Prediction
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

### Docker Deployment

1. **Build and Run**
   ```bash
   make docker-build
   make docker-run
   ```

2. **Or use Docker Compose**
   ```bash
   docker-compose up -d
   ```

3. **With Monitoring Stack**
   ```bash
   docker-compose --profile monitoring up -d
   ```

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   API Layer    │────│  Model Manager   │────│ Inference Engine │
│  (FastAPI)     │    │   (Loading)      │    │  (Optimization)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Validation    │    │   Preprocessing  │    │    Caching      │
│   & Schemas     │    │    Pipeline      │    │   & Metrics     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
carla-rl-serving/
├── src/                    # Source code
│   ├── server.py          # FastAPI application
│   ├── inference.py       # Inference engine
│   ├── model_loader.py    # Model loading utilities
│   ├── preprocessing.py   # Feature preprocessing
│   ├── io_schemas.py      # Pydantic schemas
│   ├── exceptions.py      # Custom exceptions
│   └── version.py         # Version management
├── tests/                 # Test suite
│   ├── test_*.py         # Unit tests
│   └── test_*_qa.py      # QA validation tests
├── artifacts/             # Model artifacts
│   └── v0.1.0/           # Version directory
│       ├── model.pt      # Model file
│       ├── preprocessor.pkl # Preprocessor
│       └── model_card.yaml  # Metadata
├── monitoring/            # Monitoring configuration
├── k8s/                  # Kubernetes manifests
├── scripts/              # Utility scripts
├── Dockerfile            # Container definition
├── docker-compose.yml    # Multi-service deployment
├── pyproject.toml        # Python project config
├── Makefile              # Development commands
└── README.md             # This file
```

## 🔧 API Reference

### Health & Metadata

- `GET /healthz` - Service health check
- `GET /metadata` - Model metadata and configuration
- `GET /metrics` - Prometheus metrics (optional)

### Inference

- `POST /predict` - Batch inference endpoint
- `POST /warmup` - Model warmup for optimization

### Request/Response Format

**Prediction Request:**
```json
{
  "observations": [
    {
      "speed": 25.5,
      "steering": 0.1,
      "sensors": [0.8, 0.2, 0.5, 0.9, 0.1]
    }
  ],
  "deterministic": false
}
```

**Prediction Response:**
```json
{
  "actions": [
    {
      "throttle": 0.7,
      "brake": 0.0,
      "steer": 0.1
    }
  ],
  "version": "v0.1.0",
  "timingMs": 8.5,
  "deterministic": false
}
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARTIFACT_DIR` | `artifacts` | Directory containing model artifacts |
| `MODEL_VERSION` | `v0.1.0` | Model version to load |
| `USE_GPU` | `0` | Enable GPU inference (1 for enabled) |
| `PORT` | `8080` | Server port |
| `WORKERS` | `1` | Number of uvicorn workers |
| `LOG_LEVEL` | `info` | Logging level |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |

### Model Artifacts

Each model version requires:
- `model.pt` - PyTorch/TorchScript model file
- `preprocessor.pkl` - Feature preprocessing pipeline
- `model_card.yaml` - Model metadata and configuration

**Example model_card.yaml:**
```yaml
model_name: "carla-ppo"
version: "v0.1.0"
model_type: "pytorch"
input_shape: [5]
output_shape: [3]
framework_version: "2.1.0"
description: "CarlaRL PPO policy for autonomous driving"
performance_metrics:
  reward: 850.5
  success_rate: 0.95
artifact_hashes:
  model.pt: "sha256:abc123..."
  preprocessor.pkl: "sha256:def456..."
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run integration tests
make test-integration

# Run performance tests
make test-performance

# Code quality checks
make quality
```

## 📊 Monitoring

The service exposes metrics compatible with Prometheus:

- Request latency percentiles (P50, P95, P99)
- Throughput (requests/sec, observations/sec)
- Error rates and counts
- Model version and status
- Memory usage statistics

### Grafana Dashboard

Use the included Grafana dashboard for visualization:
```bash
docker-compose --profile monitoring up -d
# Access Grafana at http://localhost:3000 (admin/admin)
```

## 🚢 Deployment

### Kubernetes

```bash
# Deploy to Kubernetes
make deploy-k8s

# Scale deployment
kubectl scale deployment carla-rl-serving --replicas=3
```

### Performance Tuning

1. **CPU Optimization**
   - Use multiple workers: `WORKERS=4`
   - Enable tensor pre-allocation
   - Use TorchScript models

2. **GPU Optimization**
   - Set `USE_GPU=1`
   - Use memory pinning
   - Batch requests for efficiency

3. **Memory Management**
   - Configure cache size appropriately
   - Monitor memory usage with `/metrics`
   - Use resource limits in containers

## 🔍 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```bash
   # Check artifact integrity
   make validate-artifacts
   
   # Verify model format
   python -c "import torch; print(torch.jit.load('artifacts/v0.1.0/model.pt'))"
   ```

2. **Performance Issues**
   ```bash
   # Run benchmark
   make benchmark
   
   # Check metrics
   curl http://localhost:8080/metrics
   ```

3. **Memory Issues**
   ```bash
   # Monitor memory usage
   docker stats carla-rl-serving
   
   # Check for memory leaks
   make test-performance
   ```

### Logs

```bash
# View container logs
docker logs carla-rl-serving

# Follow logs in real-time
docker logs -f carla-rl-serving
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test`
5. Submit a pull request

### Development Workflow

```bash
# Setup development environment
make install-dev

# Run pre-commit hooks
pre-commit run --all-files

# Run full test suite
make test-all

# Build and test Docker image
make docker-build
make docker-run
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) for the high-performance web framework
- [PyTorch](https://pytorch.org/) for the machine learning framework
- [CARLA](https://carla.org/) for the autonomous driving simulator
- [Pydantic](https://pydantic.dev/) for data validation and settings management

---

**Built with ❤️ for the autonomous driving research community**
