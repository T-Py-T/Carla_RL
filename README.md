# Highway RL

Train a deep Q-network to drive in
[`highway-env`](https://github.com/Farama-Foundation/HighwayEnv), then serve the
policy over HTTP as a versioned, hash-verified TorchScript artifact.

Two halves live here, and each is useful on its own:

- **[`model-sim/`](model-sim)** — the research half. highway-env scenarios, a DQN
  agent with optional Double DQN and dueling heads, a training loop, and
  saved-model evaluation.
- **[`model-serving/`](model-serving)** — the serving half. A FastAPI service that
  loads a versioned TorchScript policy, validates every request, and answers
  with bounded throttle, brake, and steering values.

The repository directory is named `Carla_RL` for historical reasons. The
maintained simulator in this tree is highway-env. CARLA appears only as an
optional playback path that needs a simulator you supply yourself — see
[CARLA, optional](#carla-optional).

## Why it exists

Plenty of driving-RL repositories stop at a training script: you get a loop, a
reward curve on your own screen, and a checkpoint file. The distance between
that checkpoint and something another program can actually call is where most of
the work in this repository went.

So the serving half treats a policy as a release artifact rather than a file on
disk. Every version is a directory with a model card, the preprocessor state
that produced it, and SHA-256 hashes that are verified at load time. A request
can ask for deterministic inference, which seeds Torch so the same observation
returns the same action. The service reports its own health, loaded version, git
revision, and Prometheus metrics, and it can discover, select, migrate, and roll
back between versions.

## Project status

**No training result is published in this repository.** There is no retained
training run, no evaluation report, no reward or success-rate table, and no
trained policy file anywhere in the tree. Checkpoints, logs, and generated
artifacts are gitignored, so anything you want to know about policy quality you
will have to measure yourself.

Two things that are easy to mistake for results, and are not:

- The TorchScript artifact produced by `scripts/create_example_artifacts` is a
  randomly initialized network. Its weights are not seeded, so `model.pt` hashes
  differently on every generation. It exercises the loading and request path and
  nothing else.
- The browser demo below is driven by a rule-based controller, not by a learned
  policy.

The one place in the tree with numbers from a real run is
[`AUDIT_version-updates-verify.md`](AUDIT_version-updates-verify.md), a
dependency-upgrade audit memo. Its figures are HTTP load-test latency and
throughput for the service, measured once on the author's machine against that
untrained example artifact. They say nothing about driving quality and are not a
maintained benchmark.

The two halves are also not yet joined. Training saves Keras models and the
service loads TorchScript, so a tested conversion step between them is still
missing.

## Getting started

You need **Python 3.12 or 3.13** and [**uv**](https://docs.astral.sh/uv/). Each
component has its own lockfile and virtual environment; there is no root
install step.

Not included in this repository, and required if you want the thing it gates:

| You want | You must supply |
| --- | --- |
| A trained policy to serve | Your own training run; no weights are committed |
| CARLA playback | A CARLA 0.9.15 server and its matching Python API |
| The browser demo | Node.js and npm, plus a WebGL-capable browser |

### Serve a policy

This is the fastest path to something running, and it needs no GPU, no display,
and no trained model.

```bash
cd model-serving
uv sync --locked --extra dev
uv run python -m scripts.create_example_artifacts \
  --output artifacts \
  --version v0.1.0
uv run uvicorn src.server:app --host 127.0.0.1 --port 8080
```

Interactive API documentation is then at `http://127.0.0.1:8080/docs`.

### Train an agent

```bash
cd model-sim
uv sync --locked --extra apple-gpu --extra dev
uv run python training/highway/train_highway.py \
  --scenario highway \
  --episodes 1000 \
  --double-dqn \
  --dueling-dqn \
  --no-wandb \
  --no-tensorboard
```

Scenarios are `highway`, `merge`, `intersection`, `parking`, `racetrack`, and
`curriculum`, which cycles through the other five. Drop the last two flags to
log to TensorBoard and Weights & Biases instead.

The `--extra` flag selects the TensorFlow build. `apple-gpu` is a historical
name for the plain CPU wheel, which is what you want on macOS and on Linux
without CUDA; use `nvidia-gpu` for `tensorflow[and-cuda]` on Linux with an
NVIDIA GPU. Training writes models and logs beneath `model-sim/`. Weight files
are gitignored, but the JSON metadata and config sidecars written next to them
are not, so check `git status` before you commit after a run.

Once a model exists under `models/highway`, evaluate it with:

```bash
uv run python evaluation/highway/evaluate_models.py
```

## A worked example

With the service running from [Serve a policy](#serve-a-policy), ask it what it
loaded:

```bash
curl -s http://127.0.0.1:8080/metadata
```

```json
{
  "modelName": "carla-ppo",
  "version": "v0.1.0",
  "device": "cpu",
  "inputShape": [5],
  "actionSpace": {
    "throttle": [0.0, 1.0],
    "brake": [0.0, 1.0],
    "steer": [-1.0, 1.0]
  }
}
```

Those five input features are `speed`, `steering`, and **exactly three** sensor
readings. The preprocessor shipped with the example artifact was fitted on
three-sensor observations, so sending a different number of sensors fails with
an `INFERENCE_ERROR` and a shape-mismatch message rather than a validation
error. Request an action:

```bash
curl -s -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "observations": [{
      "speed": 25.5,
      "steering": 0.1,
      "sensors": [0.8, 0.2, 0.5]
    }],
    "deterministic": true
  }'
```

```json
{
  "actions": [{"throttle": 0.46, "brake": 0.0, "steer": 0.01}],
  "version": "v0.1.0",
  "timingMs": 1.23,
  "deterministic": true
}
```

Treat those action numbers as illustrative: the example network's weights are
random, so yours will differ, and `timingMs` is whatever your machine managed.
What does hold is that repeated calls against the same artifact return identical
actions, which is the property `deterministic` exists to guarantee. One request
carries between 1 and 1,000 observations.

Two details worth knowing before you wire up a health check. `GET /healthz`
reports `degraded` until the model has been exercised, so call `POST /warmup`
first and it flips to `ok`. And `GET /versions` lists every artifact directory it
discovered along with the hashes it verified.

## Demo

The repository includes a Three.js chase-cam driving visualization under
[`model-sim/demos/web/`](model-sim/demos/web), with captured stills and video
checked in at [`docs/pr-115-artifacts/`](docs/pr-115-artifacts).

![Chase-cam view with perception overlay and driving HUD](docs/pr-115-artifacts/after_fsd_overlay.png)

Be clear about what you are looking at: the car is steered by a rule-based
controller that reacts to synthesized LiDAR and proximity readings each frame,
not by a trained network. No policy weights are loaded. What it shows is the
interface around a driving policy, not a policy.

```bash
cd model-sim/demos/web
npm install
npm run dev
```

There is also a headless OpenCV renderer that needs no browser:

```bash
cd model-sim
uv run python demos/fsd_playback.py --mode offline --headless --steps 240
```

See [3D town FSD playback](docs/carla-fsd-playback.md) for the capture and
verification scripts.

## CARLA, optional

[`model-sim/src/carla_rl/env.py`](model-sim/src/carla_rl/env.py) imports the
CARLA Python API behind a guarded import and can connect to a server, load a
town, and spawn an ego vehicle with camera and collision sensors. The
`--mode carla` branch of `demos/fsd_playback.py` uses it for live ego-camera
playback.

None of that works out of the box. The CARLA simulator is not vendored here and
cannot be; you need a CARLA 0.9.15 server, its matching Python API package, and
in practice Linux with an NVIDIA GPU.
[`model-sim/docker/setup_carla.sh`](model-sim/docker) helps fetch the server.
Highway-env training does not touch CARLA at all.

## Repository layout

| Path | Purpose |
| --- | --- |
| [`model-sim/`](model-sim) | highway-env wrapper, DQN agent, trainer, evaluation, demos, tests |
| [`model-serving/`](model-serving) | FastAPI service, artifact versioning, deployment manifests, tests |
| [`docs/`](docs) | playback notes, tooling notes, and captured demo media |
| [`.devcontainer/`](.devcontainer) | containerized development environment |
| [`Makefile`](Makefile) | shortcuts that delegate into both components |

## Validate a change

There is no hosted CI in this repository; see
[`.github/workflows/README.md`](.github/workflows/README.md) for the reasoning.
Validation is local, and each component runs in its own locked environment:

```bash
cd model-sim
uv sync --locked --extra apple-gpu --extra dev
uv run pytest tests -q

cd ../model-serving
uv sync --locked --extra dev
uv run pytest tests -q
```

From the root, `make check` runs Ruff across the tree. Its Python compile step
invokes a bare `python`, so it silently does nothing on systems where only
`python3` is on the path; the component suites above are the real gate.

## Contributing

Pull requests go against `master`. Keep a change inside the component it
touches, run that component's locked suite, and leave checkpoints, generated
artifacts, and deployment exports out of the commit. The full checklist is in
[`CONTRIBUTING.md`](CONTRIBUTING.md), and vulnerability reporting is in
[`SECURITY.md`](SECURITY.md).

## A note on the deployment files

The Dockerfile, Compose files, Kubernetes manifests, and Prometheus and Grafana
configuration under `model-serving/deploy/` are development references. The
Kubernetes example relies on `imagePullPolicy: Never`, and CORS and allowed
hosts both default to `*`. Review security, resource, ingress, and persistence
settings before running any of it outside an isolated environment.

## License

This repository is [Apache License 2.0](LICENSE) by default. The
`model-serving` component carries its own [MIT License](model-serving/LICENSE).
Third-party libraries and simulators stay under their own terms.
