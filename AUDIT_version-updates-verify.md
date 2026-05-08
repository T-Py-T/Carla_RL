# Version Updates Audit - `version-updates-verify`

**Branch:** `version-updates-verify` (forked from `origin/master` @ `45d10cb`)
**Audited:** 2026-04-18
**Scope:** Verify functionality after dependabot-driven dependency updates.
**Host:** macOS (arm64) with OrbStack. All testing performed in Linux containers (no host Python).

## 1. What was actually bumped

Tracked from the merged dependabot PRs that landed on `master`:

### `model-serving/requirements.txt` (runtime deps)

| Package | Before | After   | Impact                                    |
|---------|--------|---------|-------------------------------------------|
| torch   | 2.1.1  | 2.8.0   | **Major** — semver-minor but behavior changed |
| pytest  | 7.4.3  | 9.0.3   | **Major** — skipped 8.x, big plugin API changes |

### `model-sim/uv.lock` (transitive bumps, no code impact)

| Package    | Before  | After   |
|------------|---------|---------|
| filelock   | 3.18.0  | 3.20.3  |
| urllib3    | 2.5.0   | 2.6.3   |
| virtualenv | 20.31.2 | 20.36.1 |
| wheel      | 0.45.1  | 0.46.2  |
| pillow     | 11.3.0  | 12.2.0  |
| werkzeug   | 3.1.3   | 3.1.6   |
| pyasn1     | 0.6.1   | 0.6.3   |
| pygments   | 2.19.2  | 2.20.0  |

`model-sim/pyproject.toml` was not edited; all the above are transitives of tensorflow/wandb/virtualenv.

---

## 2. Verdict

- `model-serving` builds and runs under torch 2.8.0, **provided one real dep-update regression is fixed** (below).
- `model-sim` imports, creates envs, and runs training episodes cleanly under the bumped transitives.
- The dependency updates themselves are not "broken" — the project has a pile of pre-existing bugs that continue to exist, and **one** (`torch.load` default change) was silently introduced by the torch bump.

### Live service on torch 2.8 load test (15s, batch=1, concurrency=16):

| Metric       | Value                        |
|--------------|------------------------------|
| Total reqs   | 11,342 (100% 200 OK)         |
| RPS          | 755.6                        |
| p50 latency  | 17.5ms (HTTP e2e)            |
| p95 latency  | 22.5ms                       |
| p99 latency  | 45.0ms                       |
| Inference layer p50 (from `/metrics` histogram) | ~0.68ms |

Batch=8, concurrency=32, 15s: **9,226 reqs / 100% 200 OK / 613 RPS / 4,912 observations/sec / p50 43ms**.
Container CPU: 0.11% steady state — server is not compute bound under this workload.

---

## 3. Dep-update regression (actionable)

### **[BUG-DEP-01] `torch.load` default changed to `weights_only=True` in torch 2.6+**

**File:** `model-serving/src/model_loader.py:232`

```python
model = torch.load(str(model_path), map_location=device)
```

In torch 2.6 the default of `weights_only` changed from `False` to `True`. With `weights_only=True`,
loading a pickled `nn.Module` raises `UnpicklingError: Weights only load failed ... Unsupported global`.
Every `.pt` artifact that stores a full module (the documented "fallback" path in this loader) is now
unloadable on torch 2.8.

**Failure mode observed end-to-end:**
```
src.exceptions.ModelLoadingError: Failed to load model from /app/artifacts/v0.1.0/model.pt
torchscript_error: PytorchStreamReader failed locating file constants.pkl: file not found
pytorch_error: Weights only load failed. ... WeightsUnpickler error: Unsupported global:
  GLOBAL __main__.ExampleCarlaModel was not an allowed global by default.
```

**Fix:** pass `weights_only=False` explicitly (backwards-compatible, restores pre-2.6 semantics):

```python
model = torch.load(str(model_path), map_location=device, weights_only=False)
```

Or (preferred, more secure): save artifacts as TorchScript via `torch.jit.save` and/or as state_dict-only
with a known architecture class, and load with `weights_only=True` everywhere.

---

## 4. Pre-existing bugs exposed by the audit (not caused by the dep bumps)

These were broken on the old torch 2.1.1 / pytest 7.4.3 too — confirmed by running the identical checks
on both images. The dep bumps didn't change them; they just became more visible.

### `model-serving`

#### [PRE-01] `SemanticVersion` dataclass conflicts with manual `__lt__`
`model-serving/src/versioning/semantic_version.py`
```
TypeError: Cannot overwrite attribute __lt__ in class SemanticVersion.
Consider using functools.total_ordering
```
`@dataclass(frozen=True, order=True)` auto-generates `__lt__`/`__le__`/`__gt__`/`__ge__`, but the class also
defines these manually. Blocks any import of `src.versioning`, which blocks the server startup path that
reaches `VersionSelector`.
**Fix:** drop `order=True` and use `@functools.total_ordering` with the manual comparison methods.

#### [PRE-02] `VersionSelectionStrategy.LATEST_STABLE` is referenced but not defined
`model-serving/src/versioning/version_selector.py`
The selector calls `VersionSelectionStrategy.LATEST_STABLE` in the stable-version code path, but the enum
only defines `LATEST`, `EXACT`, `COMPATIBLE`, `RANGE`. Every call site that hits this path raises
`AttributeError`.
**Fix:** add `LATEST_STABLE = "stable"` (or whichever string is expected) to the enum.

#### [PRE-03] `VersionSelectionStrategy` isn't exported from `src.versioning`
`model-serving/src/versioning/__init__.py`
Only `SemanticVersion` and `VersionSelector` are re-exported, but `src.server` imports
`VersionSelectionStrategy` from `src.versioning`. Raises `ImportError` on startup.
**Fix:** add `VersionSelectionStrategy` to `__all__` and the import list.

#### [PRE-04] `Span` isn't a context manager despite `with tracer.trace_*:` usage
`model-serving/src/monitoring/tracing.py`
`tracer.trace_model_loading()` / `trace_model_warmup()` return `Span` objects. `server.py` uses them as
`with tracer.trace_...(): ...` context managers, but `Span` has no `__enter__`/`__exit__`. Raises
`TypeError: 'Span' object does not support the context manager protocol`.
**Fix:** implement `__enter__` (returns self) and `__exit__` (calls `self.finish(...)` with success/error
status) on `Span`.

#### [PRE-05] Runtime deps missing from `requirements.txt`
`model-serving/requirements.txt` vs `model-serving/src/` actual imports:

| Package            | Used in                                        | In requirements.txt? |
|--------------------|------------------------------------------------|----------------------|
| `prometheus_client`| `src/monitoring/metrics.py`, `src/server.py`   | No                   |
| `psutil`           | `src/optimization/*`, `src/benchmarking/*`     | No                   |
| `jinja2`           | `src/versioning/...`                           | No                   |
| `watchdog`         | `src/versioning/artifact_manager.py`           | No                   |
| `deepdiff`         | `src/versioning/compatibility.py`              | No                   |
| `requests`         | `src/versioning/integrity_validator.py`        | No                   |

The Dockerfile currently builds from `requirements.txt` alone, so a container built from the repo as-is
has `ImportError` on every one of these modules on `src.server` startup. `pyproject.toml` lists them in
`[project.optional-dependencies.monitoring]` and elsewhere, but the Dockerfile never installs the extras.
**Fix:** either move these to `requirements.txt`, or change the Dockerfile to `pip install -e .[monitoring]` (etc.).

#### [PRE-06] `PolicyWrapper.forward` passes `deterministic` to TorchScript models
`model-serving/src/model_loader.py:70-85`
For `model_type == "torchscript"`, every scripted model has a `forward` attribute (it's a method on
`nn.Module`), so the `hasattr(self.model, "forward")` branch always wins and calls
`self.model.forward(x, deterministic)`. Scripted `forward(self, x)` only accepts one tensor arg, so
every call into a scripted model raises:
```
RuntimeError: forward() expected at most 2 argument(s) but received 3 argument(s)
```
This is also what makes `/warmup` fail unless the artifact is a pickled `nn.Module` (which then hits [BUG-DEP-01]).
**Fix:** inspect the scripted module's forward signature, or standardise on the `.act(x, deterministic)`
convention and only fall through to `forward(x)` without the extra arg.

#### [PRE-07] Warmup / test fixtures use wrong sensor dimensionality
`model-serving/src/server.py:372`, `model-serving/src/inference.py:425`, and most `tests/test_inference*.py`
fixtures construct `Observation(..., sensors=[0.5] * 5)` (7-dim input), but the preprocessor is fit for
3-sensor observations (5-dim input — `speed`, `steering`, 3 sensors). Warmup and most `test_inference_*`
cases fail with:
```
mat1 and mat2 shapes cannot be multiplied (1x7 and 5x64)
```
The `/predict` path works fine with correctly-shaped real observations. **Fix:** use
`sensors=[0.5, 0.5, 0.5]` in warmup and the affected test fixtures.

#### [PRE-08] `model_card.yaml` has placeholder hashes
`model-serving/artifacts/v0.1.0/model_card.yaml`
```yaml
files:
  model.pt: "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456"
  preprocessor.pkl: "b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef1234567a"
```
Those are literal fake hex. The integrity validator raises `ArtifactValidationError: Hash mismatch for model.pt`
on startup, which `src.server` catches and converts to a model-load failure — server comes up "unhealthy".
**Fix:** `scripts/create_example_artifacts.py` already computes correct hashes; run it and commit (or re-run
on CI and bake into the image). Alternatively skip hash validation when both sides are the pristine repo artifact.

#### [PRE-09] `scripts/create_example_artifacts.py` has a broken import setup
`model-serving/scripts/create_example_artifacts.py`
Top of the file does `sys.path.insert(0, 'src')` then `from preprocessing import ...`, but `preprocessing.py`
itself uses relative imports (`from .exceptions import ...`). Running the script verbatim raises
`ImportError: attempted relative import with no known parent package`. **Fix:** either import via the
`src.` namespace (`from src.preprocessing import ...`) or make `src/` an installable package.

#### [PRE-10] `tests/versioning/test_semantic_version.py` imports wrong package path
```
from model_serving.src.versioning.semantic_version import ...
ModuleNotFoundError: No module named 'model_serving'
```
Entire `tests/versioning/` folder fails pytest collection on the new pytest 9 because of this.
**Fix:** `from src.versioning.semantic_version import ...`.

#### [PRE-11] `tests/test_preprocessing.py::test_to_feature_matrix` float equality
```python
assert features[1, 1] == 0.1    # AssertionError: assert 0.1 == 0.1
```
`to_feature_matrix` casts to `float32`, so the literal Python float `0.1` doesn't compare equal.
**Fix:** `np.isclose(features[1, 1], 0.1)` or assert on `float32(0.1)`.

#### [PRE-12] `uvicorn --log-level` and Python `LOG_LEVEL` case convention mismatch
`model-serving/entrypoint.sh` (generated by Dockerfile) reads `LOG_LEVEL` and calls
`uvicorn --log-level "$LOG_LEVEL"`. uvicorn's CLI expects **lowercase** (`info`), but the application's
structured logging expects **uppercase** (`INFO`). Whatever value the user sets, one side errors:
- `LOG_LEVEL=info` → Python `logging`: `ValueError: Unknown level: 'info'`
- `LOG_LEVEL=INFO` → uvicorn CLI: `Invalid value for '--log-level'`

**Fix:** in `entrypoint.sh`, normalize: `exec uvicorn ... --log-level "$(echo "${LOG_LEVEL:-INFO}" | tr '[:upper:]' '[:lower:]')"`,
and export the uppercase form to Python separately.

### `model-sim`

#### [PRE-13] `environment.py` forgets to `import highway_env`
`model-sim/src/highway_rl/environment.py`
Imports `gymnasium as gym` but not `highway_env`. highway-env 1.10 does not register its envs via
`gymnasium.envs` entry_points (no entry_points declared in the installed metadata), so custom env IDs
like `highway-fast-v0` are only registered when something explicitly runs `import highway_env`. Running
`HighwayEnvironment('highway')` in a fresh interpreter raises:
```
gymnasium.error.NameNotFound: Environment `highway-fast` doesn't exist.
```
The Makefile's `validate` target papers over this by including `import highway_env` in the one-liner.
**Fix:** add `import highway_env  # noqa: F401` at the top of `environment.py`.

#### [PRE-14] `tests/test_cross_platform.py` has stale package/arch assumptions
- Asserts `machine in ['x86_64', 'arm64', ...]` — fails on Linux aarch64 (which is what containers report).
- Imports `carla_rl` — that package doesn't exist in this repo; the sim package is `highway_rl`.
**Fix:** update arch allowlist to include `aarch64`; update imports to `highway_rl`.

---

## 5. How the audit was run

1. `git checkout version-updates-verify` (branched from `origin/master`).
2. Built `carla-rl-serving:audit` from `model-serving/Dockerfile` (production target) — confirms
   `torch 2.8.0+cpu`, `pytest 9.0.3`, everything else per `requirements.txt`.
3. Built `carla-rl-serving:audit-full` layered on top with the optional-deps that the code actually
   needs (`prometheus-client`, `psutil`, `jinja2`, `watchdog`, `deepdiff`, `requests`, `pytest-cov`).
4. Generated fresh artifacts (`model.pt` + `preprocessor.pkl` + corrected-hash `model_card.yaml`)
   in `/tmp/carla_artifacts/v0.1.0/` via a wrapper around `scripts/create_example_artifacts.py`.
   Artifacts are bind-mounted, not committed (`.gitignore` now excludes `*.pt`, `*.pkl`, `*.pth`, `*.onnx`).
5. Bind-mounted patched versions of the files hit by PRE-01..04 and BUG-DEP-01 at container runtime.
   No tracked source files were modified — all fixes were overlay mounts so the audit could exercise
   the code path without pretending to fix bugs.
6. Started the service via `docker run ... uvicorn src.server:app`. Confirmed `Up (healthy)` and
   `"Model loaded successfully"` log line.
7. Exercised `/healthz`, `/metadata`, `/warmup`, `/predict` (single + batch=4) via curl.
8. Ran the custom HTTP load test (`/tmp/carla_load_test.py`) against `/predict`.
9. Ran `pytest` inside the container: 351 passed / 62 failed (excluding the known-broken `tests/versioning`
   and integration tests requiring an external server). Every failure I sampled maps to an item
   in §4.
10. Built `carla-rl-sim:audit` (python:3.11-slim + CPU-only `tensorflow 2.15.1` + deps from
    `model-sim/pyproject.toml` minus Apple-Silicon extras). Confirmed all dependabot transitives load
    cleanly. Ran `tests/test_cross_platform.py` (5 pass / 4 pre-existing fail per §4). Ran a 3-episode
    highway-env DQN smoke, which completed successfully.
11. Stopped containers. `docker images | grep carla-rl` retains the built images for re-runs; no new
    files were left behind in the workspace (verified with `git status`).

---

## 6. Recommended next steps

Priority order:

1. **Fix BUG-DEP-01** (`torch.load` default change) — this is the one regression actually caused by the
   dependabot bumps. One line. Unblocks all `nn.Module`-pickle artifacts.
2. Fix PRE-05 (requirements.txt missing deps) — without it the production Dockerfile image can't even
   start the server. This is the highest-impact pre-existing bug.
3. Fix PRE-01..04 (versioning + tracing patches) — needed for `/healthz`-style model loading to succeed
   on a clean build; currently only works because patches are overlaid.
4. Fix PRE-08 (placeholder hashes in `model_card.yaml`) — needed if the integrity validator is meant to
   actually catch anything; either populate real hashes or disable the check for bundled artifacts.
5. Fix PRE-07 (wrong-dim warmup fixture) — so `/warmup` works without errors.
6. Fix PRE-12 (log level casing) — only affects operators setting `LOG_LEVEL`.
7. Fix PRE-13 (`import highway_env`) — unblocks anyone instantiating `HighwayEnvironment` directly.
8. Clean up tests (PRE-10, PRE-11, PRE-14) — low impact, but the test suite is currently misleading.

None of these require a dependency downgrade. The dep bumps themselves are fine.
