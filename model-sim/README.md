# Highway RL simulation

`model-sim` trains a DQN driving policy against the discrete-action scenarios
provided by `highway-env`. It supports one scenario at a time or a curriculum
that cycles through highway, merge, intersection, parking, and racetrack
environments.

## Agent and environment

The `HighwayEnvironment` wrapper normalizes Gymnasium reset and step behavior,
selects the scenario, and records episode statistics. `HighwayDQNAgent`
provides replay memory, a target network, epsilon-greedy action selection,
optional Double DQN targets, and an optional dueling network head.

```text
highway-env observation
        │
        ▼
DQN / dueling DQN
        │ action
        ▼
environment step ──► reward, terminal state, episode statistics
        │
        └──────────► replay buffer and target-network updates
```

## Setup

Requirements: Python 3.12 or 3.13 and `uv`.

```bash
uv sync --locked --extra apple-gpu --extra dev
```

The historical extra name `apple-gpu` installs the TensorFlow dependency used
by the simulator. Current TensorFlow releases on Apple Silicon use the standard
macOS wheel; this setup does not promise Metal acceleration.

## Train

```bash
uv run python training/highway/train_highway.py \
  --scenario highway \
  --episodes 1000 \
  --double-dqn \
  --dueling-dqn
```

Available scenarios are `highway`, `merge`, `intersection`, `parking`,
`racetrack`, and `curriculum`. Use `--no-wandb --no-tensorboard` for a local run
without external experiment tracking.

Generated models, checkpoints, logs, and benchmark output remain local and are
ignored by Git.

## Evaluate

After training has written a model beneath `models/highway`:

```bash
uv run python evaluation/highway/evaluate_models.py
```

Evaluation output is meaningful only with the matching model, scenario
configuration, dependency lock, and random-seed policy. No result bundle is
currently published in the repository.

## Test

```bash
uv run pytest tests -q
uv run ruff check src scripts training evaluation tests
```

The tests include trainer behavior and platform setup checks. They do not train
a converged policy.

## Layout

| Path | Purpose |
| --- | --- |
| `src/highway_rl/` | environment, agent, trainer, and logging modules |
| `training/highway/` | command-line training entry point |
| `evaluation/highway/` | saved-model evaluation |
| `scripts/` | platform inspection, playback, and benchmark helpers |
| `tests/` | behavioral, validation, and benchmark tests |
