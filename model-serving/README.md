# Driving policy service

A FastAPI service for loading a versioned TorchScript driving policy and
turning batches of vehicle observations into throttle, brake, and steering
actions.

The service is a reusable inference boundary. It validates requests, verifies
artifact hashes, exposes model and health metadata, records Prometheus metrics,
and keeps model-version management separate from the HTTP layer.

## Request path

```text
JSON observations
      │
      ▼
Pydantic validation ──► preprocessing ──► TorchScript policy
                                               │
                                               ▼
                                    bounded action response
                                               │
                                               ├── timing and request ID
                                               └── metrics and structured logs
```

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /healthz` | Service, model, version, device, and Git state |
| `GET /metadata` | Model name, input shape, and action bounds |
| `POST /predict` | One to 1,000 observations per request |
| `POST /warmup` | Exercise the loaded model before traffic |
| `GET /metrics` | Prometheus-formatted process and inference metrics |
| `GET /versions` | Discover versioned artifact directories |

Interactive OpenAPI documentation is available at `/docs` while the service is
running.

## Quick start

Requirements: Python 3.12 or 3.13 and `uv`.

```bash
uv sync --locked --extra dev
uv run python -m scripts.create_example_artifacts \
  --output artifacts \
  --version v0.1.0
uv run uvicorn src.server:app --host 127.0.0.1 --port 8080
```

The generator creates a small local TorchScript artifact for exercising the
service. It is not a trained driving model.

Send a deterministic prediction request:

```bash
curl --request POST http://127.0.0.1:8080/predict \
  --header 'Content-Type: application/json' \
  --data '{
    "observations": [{
      "speed": 25.5,
      "steering": 0.1,
      "sensors": [0.8, 0.2, 0.5, 0.9, 0.1]
    }],
    "deterministic": true
  }'
```

## Artifact contract

Each version is a directory such as `artifacts/v0.1.0/` containing:

| File | Purpose |
| --- | --- |
| `model.pt` | TorchScript policy loaded with `torch.jit.load` |
| `preprocessor.json` or `preprocessor.pkl` | Feature normalization state |
| `model_card.yaml` | Model name, semantic version, type, shapes, and file hashes |

Artifact directories are generated and ignored by Git. See
[Artifact lifecycle](docs/artifacts.md) for validation and publication rules.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `ARTIFACT_DIR` | `artifacts` | Root containing version directories |
| `MODEL_VERSION` | `v0.1.0` | Static fallback version |
| `USE_GPU` | `0` | Select CUDA when it is available |
| `PORT` | `8080` in the container | HTTP port used by the entrypoint |
| `WORKERS` | `1` | Uvicorn worker count |
| `LOG_LEVEL` | `info` | Service log level |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `ALLOWED_HOSTS` | `*` | Comma-separated trusted hosts |

Restrict CORS and host values before exposing the service outside a local
environment.

## Validate a change

```bash
uv run pytest tests -q
uv run ruff check src scripts tests
uv run mypy src
```

The suite covers schemas, inference, caching, API behavior, artifact integrity,
version selection, configuration helpers, monitoring, and benchmark mechanics.
It does not establish the quality of a trained policy. See
[Testing](docs/testing.md) and [Benchmarking](docs/benchmarking.md).

## Containers and Kubernetes

The Dockerfile builds an unprivileged CPU image and generates the example
artifact during the build. Compose adds Prometheus and Grafana. The Kubernetes
manifest is a local development example that uses `imagePullPolicy: Never`.

See [Deployment](docs/deployment.md) before running either path.

## Current boundaries

- No trained driving policy is published in this directory.
- The example artifact proves the serving path only.
- The service loads TorchScript; the sibling simulation component saves Keras
  models, so an explicit, tested conversion step is still required.
- Performance depends on the model, batch shape, hardware, and process settings.
  No benchmark number is asserted in this README.
- The deployment files need environment-specific security review before shared or
  production use.

## License

This component is available under the [MIT License](LICENSE).
