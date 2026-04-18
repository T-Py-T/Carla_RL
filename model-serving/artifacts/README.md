# `model-serving/artifacts/`

This directory holds example model artifacts consumed by the CarlaRL policy
service at runtime (`v<semver>/` subfolders containing `model.pt`,
`preprocessor.pkl`, and `model_card.yaml`).

The individual version folders are **generated, not source of truth** — they
are rebuilt from `scripts/create_example_artifacts.py` every time the
production Docker image is built or the devcontainer is (re)created, and the
`model_card.yaml` file embeds SHA-256 hashes of the binaries that only match
after a fresh regeneration. Committing them would produce noisy binary /
hash-only diffs on every rebuild.

## Regenerating locally

```bash
cd model-serving
python -m scripts.create_example_artifacts --output artifacts --version v0.1.0
```

The `.devcontainer/setup.sh` post-create hook performs this automatically when
the `v0.1.0/` folder is missing, so freshly-cloned checkouts work without any
manual step.

## Why the `v*/` folders are git-ignored

- `.pt` / `.pkl` match the global `*.pt` / `*.pkl` rules in the repo-root
  `.gitignore`.
- `model_card.yaml` lives inside `model-serving/artifacts/v*/` which is
  explicitly ignored there, so the whole version folder stays out of history.

If you need to ship a specific pre-built artifact set (e.g. a trained policy
from a CI release job), publish it as a release asset or push to an artifact
store — do not commit the bytes into this folder.
