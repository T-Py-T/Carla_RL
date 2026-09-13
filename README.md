# Highway RL Platform

A reinforcement-learning workspace for training driving policies in
[`highway-env`](https://github.com/Farama-Foundation/HighwayEnv) and serving
versioned policy artifacts through an HTTP API.

The repository has two independent components. `model-sim` implements the
simulation, DQN agent, training loop, and evaluation tools. `model-serving`
implements a FastAPI service for validated TorchScript artifacts, with batch
inference, health checks, metrics, version selection, and container deployment.

## How it fits together

```text
highway-env scenarios
        │
        ▼
Keras DQN agent ──► checkpoints and evaluation output
        │
        │ explicit model conversion is still required
        ▼
versioned TorchScript artifact + model card
        │
        ▼
FastAPI policy service ──► predictions, health, metadata, metrics
```

The simulation and serving code do not currently share an automatic export
pipeline. The checked-in serving example generates a small test artifact; it is
not a trained driving policy.

## Implemented components

### Simulation and training

- `highway`, `merge`, `intersection`, `parking`, and `racetrack` scenarios
- dense and image DQN networks with optional Double DQN and dueling heads
- replay memory, target-network updates, epsilon-greedy exploration, and model
  checkpoints
- single-scenario and curriculum training entry points
- TensorBoard and optional Weights & Biases logging
- trainer and cross-platform regression tests

### Policy service

- validated request and response schemas for single or batched observations
- TorchScript loading with SHA-256 artifact checks
- deterministic inference, bounded batching, in-memory caching, and warmup
- `/healthz`, `/metadata`, `/predict`, `/warmup`, `/metrics`, and `/versions`
- model version discovery, content-addressed storage, migration, and rollback
  helpers
- Docker, Compose, Kubernetes, Prometheus, and Grafana configuration

## Run the simulator

Requirements: Python 3.12 or 3.13 and [`uv`](https://docs.astral.sh/uv/).

```bash
cd model-sim
uv sync --locked --extra apple-gpu --extra dev
uv run python training/highway/train_highway.py \
  --scenario highway \
  --episodes 1000 \
  --double-dqn \
  --dueling-dqn
```

Use `--scenario curriculum` to cycle through all five environments. Training
writes checkpoints and logs beneath `model-sim/`; those generated files are not
committed.

See [`model-sim/README.md`](model-sim/README.md) for evaluation and platform
notes.

## Run the policy service

Generate the repository's example artifact, then start the API:

```bash
cd model-serving
uv sync --locked --extra dev
uv run python -m scripts.create_example_artifacts \
  --output artifacts \
  --version v0.1.0
uv run uvicorn src.server:app --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080/docs` for the interactive API schema. The example
artifact proves the loading and request path; it is not evidence of driving
quality.

See [`model-serving/README.md`](model-serving/README.md) for the artifact
contract, API example, configuration, and container workflow.

## Validate a change

Run the component suites in their own locked environments:

```bash
cd model-sim
uv sync --locked --extra apple-gpu --extra dev
uv run pytest tests -q

cd ../model-serving
uv sync --locked --extra dev
uv run pytest tests -q
```

The repository also provides `make check` at the root for Python compilation
and Ruff checks. No GitHub-hosted workflow is configured; validation is local.

## Repository layout

| Path | Purpose |
| --- | --- |
| [`model-sim/`](model-sim) | highway-env wrapper, DQN agent, trainer, evaluation, and tests |
| [`model-serving/`](model-serving) | FastAPI policy service, artifact management, deployment, and tests |
| [`.devcontainer/`](.devcontainer) | containerized development environment |
| [`Makefile`](Makefile) | top-level shortcuts for both components |

## Current boundaries

- No trained model, training dataset, or retained evaluation result is published
  in this repository.
- Simulation checkpoints are Keras models; the service consumes TorchScript.
  A tested conversion step is still needed to connect them.
- The historical CARLA path remains secondary. The maintained simulator in this
  tree is `highway-env`.
- Performance tools and thresholds are available, but measured service results
  depend on the selected artifact and hardware and are not stated here.
- Docker and Kubernetes files are development references. Review security,
  resource, ingress, and persistence settings before using them outside an
  isolated environment.

## License

The original repository is licensed under Apache License 2.0. The
`model-serving` component includes its own [MIT License](model-serving/LICENSE).
Third-party libraries and simulators remain under their respective terms.
