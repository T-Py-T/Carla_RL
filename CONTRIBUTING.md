# Contributing (lean)

Propose changes via pull request against `master`. This repository has no
hosted CI workflow; reviewers rely on local checks described in the root
README.

## Before you open a PR

1. Scope changes to the component you touched (`model-sim/`, `model-serving/`,
   or root docs/tooling).
2. Run the locked `uv` + `pytest` suites for affected components (see
   [Validate a change](README.md#validate-a-change) in [`README.md`](README.md)).
3. From the repository root, `make check` runs Python compile and Ruff checks.
4. Do not commit secrets, checkpoints, personal data, or unredacted deployment
   exports. See [`SECURITY.md`](SECURITY.md).

## Docs and discoverability

- [`SECURITY.md`](SECURITY.md) — vulnerability reporting and repository
  boundaries
- [`LICENSE`](LICENSE) — Apache 2.0 default; [`model-serving/LICENSE`](model-serving/LICENSE)
  is MIT for that component

## Snapshot cite

Orientation for this docs slice only—not a release gate, visual gate, score, or
“READY” claim.

| Field | Value |
| --- | --- |
| `master` tip prefix at authoring | `c002379c` |
| Steward resolve | Pending on the opening PR for Ship 258 |

After merge, re-check `git rev-parse origin/master`; the tip moves independently
of this file.
