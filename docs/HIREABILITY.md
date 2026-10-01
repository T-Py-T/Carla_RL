# Hireability and discoverability (lean index)

This page orients reviewers, hiring managers, and search tools on **Carla_RL**
without claiming release readiness, benchmark scores, or authorization to
operate a production fleet.

## What

**Carla_RL** (root Python meta-package `highway-rl`) is a reinforcement-learning
workspace with two maintained components:

- **`model-sim`** — DQN training and evaluation on
  [`highway-env`](https://github.com/Farama-Foundation/HighwayEnv) scenarios
- **`model-serving`** — FastAPI service for validated TorchScript policy
  artifacts (health, metadata, batch inference, versioning helpers)

Experimental CARLA-related helpers and local 3D playback demos exist but are
secondary; the maintained simulator path in this tree is `highway-env`. See
[Current boundaries](../README.md#current-boundaries) in the root README.

## Why

Training and serving are separated so you can iterate on policies locally,
convert artifacts explicitly, and exercise inference through a schema-checked
HTTP API—while keeping clear limits on what is checked in (no published trained
weights or retained evaluation scores in this repository).

## How to navigate

| Goal | Document |
| --- | --- |
| Train or evaluate policies | [`model-sim/README.md`](../model-sim/README.md) |
| Run or containerize the API | [`model-serving/README.md`](../model-serving/README.md) |
| Report a vulnerability | [`SECURITY.md`](../SECURITY.md) |
| 3D FSD-style playback notes | [`docs/carla-fsd-playback.md`](carla-fsd-playback.md) |
| Python / tooling notes | [`docs/python-three-shim.md`](python-three-shim.md) |
| Serving deep dives | [`model-serving/docs/`](../model-serving/docs/) |
| Root runbooks and layout | [`README.md`](../README.md) |

Validate changes with the locked `uv` environments and pytest suites in the
root README; `make check` runs root-level Ruff and compile checks.

There is no separate `CONTRIBUTING.md`; propose changes via pull request against
`master` and keep docs accurate about local-only validation.

## Suggested GitHub topics

For repository discoverability only (not quality endorsements):

`reinforcement-learning` `deep-q-network` `highway-env` `gymnasium` `pytorch`
`torchscript` `fastapi` `model-serving` `autonomous-driving` `simulation`

## License

- Repository default: [Apache License 2.0](../LICENSE)
- `model-serving` component: [MIT](../model-serving/LICENSE)

Third-party simulators, web demo assets, and Python dependencies remain under
their respective licenses.

## Snapshot cite (tip≠READY)

Orientation for this docs slice only—not a release gate, visual gate, score, or
“READY” claim.

| Field | Value |
| --- | --- |
| `master` tip prefix at authoring | `b03d04c` |
| Steward resolve | Pending on the opening PR for Ship 254 |

After merge, re-check `git rev-parse origin/master`; the tip moves independently
of this file.
