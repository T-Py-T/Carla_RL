# Testing

Install the locked development environment:

```bash
uv sync --locked --extra dev
```

Run the complete unit and integration suite:

```bash
uv run pytest tests -q
```

Run focused areas while changing one subsystem:

```bash
uv run pytest tests/test_schemas.py tests/test_api_endpoints.py -q
uv run pytest tests/test_inference.py -q
uv run pytest tests/versioning -q
uv run pytest tests/config -q
uv run pytest tests/monitoring -q
uv run pytest tests/benchmarking -q
```

Static checks:

```bash
uv run ruff check src scripts tests
uv run mypy src
```

Most tests construct fake policies, temporary artifact directories, or an
in-process FastAPI client. They validate the service contract without a trained
model or external network. Deployment scripts under `scripts/` contain separate
checks that expect a running container or Kubernetes service.

When a test uses a fake inference engine or synthetic artifact, report it as a
service-path test rather than a model-quality result.
