# Contributing

Propose changes via pull request against `master`. This repository has no
hosted CI workflow; reviewers rely on local checks described in the root
README.

## Before you open a PR

1. Scope changes to the component you touched (`model-sim/`, `model-serving/`,
   or root docs/tooling).
2. Run the locked `uv` + `pytest` suites for affected components (see
   [Getting started](README.md#getting-started) in [`README.md`](README.md)).
3. From the repository root, `make check` runs Python compile and Ruff checks.
4. Do not commit secrets, checkpoints, personal data, or unredacted deployment
   exports. See [`SECURITY.md`](SECURITY.md).

## Related docs

- [`SECURITY.md`](SECURITY.md) — vulnerability reporting and repository
  boundaries
- [`LICENSE`](LICENSE) — Apache 2.0 default; [`model-serving/LICENSE`](model-serving/LICENSE)
  is MIT for that component

## Pull requests

Keep each pull request focused on one change, describe what you tested in the
pull request template, and expect review feedback before merge.
