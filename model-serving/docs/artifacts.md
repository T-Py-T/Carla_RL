# Artifact lifecycle

The service discovers model versions beneath `ARTIFACT_DIR`. Each immediate
subdirectory must use a semantic version such as `v0.1.0` and contain the model,
preprocessor state, and model card needed for one immutable policy release.

## Create the example artifact

```bash
uv run python -m scripts.create_example_artifacts \
  --output artifacts \
  --version v0.1.0
```

This command generates a small test policy and its metadata. It is useful for
API and container checks, but it is not a trained driving policy.

## Validate a version

```bash
uv run python scripts/validate_artifacts.py \
  --artifact-dir artifacts/v0.1.0
```

The loader checks required model-card fields and compares each recorded SHA-256
digest with the corresponding artifact file. TorchScript is the supported safe
loading path. A raw PyTorch state dictionary needs an application-owned model
architecture and is rejected by the generic loader.

## Publish a trained policy

1. Export the trained policy to TorchScript in an isolated conversion step.
2. Run the policy against known observations before packaging it.
3. Write a model card with the model name, version, type, input shape, action
   shape, training source revision, and applicable limitations.
4. Calculate hashes after every file is final.
5. Validate the directory with the command above.
6. Publish the directory as an immutable release asset or to an artifact store.

Do not commit generated `.pt`, `.pkl`, or version directories. Do not load an
untrusted pickle or Python checkpoint in the service process.

## Version selection

At startup, the service scans the artifact root and selects the version defined
by its configured strategy. `MODEL_VERSION` remains the static fallback. Use
`GET /versions` to inspect what was discovered and which version is active.
