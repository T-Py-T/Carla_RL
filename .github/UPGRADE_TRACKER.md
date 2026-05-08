# Python and dependency upgrade tracker

This file is the **hand-off list** for future agents. Do **not** fold unrelated major bumps into routine PRs: each breaking area gets its **own branch** and its own review/CI cycle.

## Current baseline (what `master` does today)

| Area | Python | How it is enforced |
|------|--------|--------------------|
| **Root** (`/pyproject.toml`) | **3.14+** | `requires-python >= 3.14`. CI: `uv sync --dev --python 3.14`. |
| **`model-serving`** | **3.14+** | `requires-python >= 3.14`, Docker / devcontainer `python:3.14-slim`, `uv lock` on **3.14**, `uv export` → `requirements.txt`. CI: `uv sync --extra dev --python 3.14`. |
| **`model-sim`** | **Declared `>=3.10,<3.16`** | **TensorFlow 2.21.0** has wheels only up to **`cp313`** (no `cp314` yet). So **`model-sim/uv.lock` is generated with `--python 3.13`**, and **merge-validation** runs `uv sync --extra dev --python 3.13` in `model-sim/`. Ruff/mypy metadata in `model-sim/pyproject.toml` targets **3.13** to match that lock. |

### Lockfile commands (copy-paste)

```bash
# Root + model-serving (3.14)
(cd /workspace && uv lock --upgrade --python 3.14)
(cd /workspace/model-serving && uv lock --upgrade --python 3.14 && uv export --no-dev --no-editable --no-emit-package carla-rl-serving --no-hashes --format requirements-txt -o requirements.txt)

# model-sim — keep 3.13 until TensorFlow ships 3.14 wheels
(cd /workspace/model-sim && uv lock --upgrade --python 3.14)   # fails today: no tensorflow cp314
(cd /workspace/model-sim && uv lock --upgrade --python 3.13)   # use this until blocker cleared
```

---

## Own-branch items (breaking / high-risk)

Use **one branch per row**. Names are suggestions only (no vendor prefixes).

### 0. Unify `model-sim` on Python 3.14 (blocked upstream)

- **Branch idea:** `upgrade/model-sim-python-3-14`
- **Trigger:** TensorFlow (or the TF version you pin) publishes **`cp314`** wheels on PyPI **or** you replace TF with a stack that supports 3.14.
- **Scope:** `uv lock --upgrade --python 3.14` in `model-sim/`; bump `[tool.ruff]` / `[tool.mypy]` to `3.14`; change merge-validation `model-sim` step to `--python 3.14`; full `pytest tests/`.
- **Exit:** Same CI Python for all three trees, or document intentional split if one tool stays behind.

### 1. Pydantic v3 + FastAPI compatibility

- **Branch idea:** `upgrade/pydantic-v3-fastapi`
- **Scope:** Remove `pydantic>=2.x,<3` caps in `model-serving/pyproject.toml`; upgrade FastAPI stack to versions that officially support Pydantic v3; fix `src/` and tests.
- **Exit:** `uv sync --extra dev --python 3.14`, full `pytest`, optional `mypy` if enabled.
- **Blocked by:** FastAPI release line and migration notes.

### 2. TensorFlow beyond 2.21.x

- **Branch idea:** `upgrade/tensorflow-next`
- **Scope:** Relax `tensorflow>=2.21,<2.22` in `model-sim` extras and dev; re-lock on **3.13** (or 3.14 when row **0** is done); run `model-sim` tests and training smoke scripts.
- **Exit:** CPU and (if applicable) `tensorflow[and-cuda]` resolution on Linux CI.

### 3. PyTorch 3.x (when you adopt it)

- **Branch idea:** `upgrade/torch-3`
- **Scope:** Relax `torch>=2.x,<3` in `model-serving`; regenerate `requirements.txt` and Docker build; run serving tests + artifact script.
- **Exit:** Docker image build and inference tests on CPU (and GPU job if you have one).

### 4. Mypy 2 strictness / typing cleanup

- **Branch idea:** `upgrade/mypy-strict` or `chore/mypy-fixes`
- **Scope:** Fix `model-serving` / `model-sim` `src/` until `mypy` passes under existing strict settings; optionally add mypy to merge-validation.
- **Exit:** Document mypy gate in CI if you add it.

### 5. Raise `model-sim` minimum Python (optional)

- **Branch idea:** `upgrade/model-sim-min-python-3-11` (or `3.12`)
- **Scope:** Increase `requires-python` lower bound once you drop older interpreters; simplify `uv.lock` markers; update docs.
- **Exit:** Align with TensorFlow’s documented minimum for the TF version you pin.

### 6. Pre-commit hook pin refresh

- **Branch idea:** `chore/pre-commit-bumps`
- **Scope:** Update `rev:` pins in `.pre-commit-config.yaml`; run `pre-commit run --all-files`.
- **Exit:** No hook drift vs `pyproject` tool versions.

### 7. Python 3.15+ (when stable and wheels exist)

- **Branch idea:** `upgrade/python-3-15`
- **Scope:** Bump `requires-python`, CI `python-version`, Docker base tags; re-lock all projects; extend merge-validation `model-sim` `--python` when TF supports it.
- **Exit:** Full test matrix on one CI image.

---

## Agent checklist (any upgrade branch)

1. One concern per branch; keep diffs reviewable.
2. After editing `pyproject.toml`, re-lock with the **correct `--python`** per tree (see **Lockfile commands** above).
3. `model-serving`: run `uv export ... -o requirements.txt` when runtime deps change.
4. Run **model-serving** `pytest` on **3.14** and **model-sim** `pytest` on the **same minor** CI uses for `model-sim` (today: **3.13**).
5. If Docker changes: `docker build -f model-serving/Dockerfile model-serving/` locally when possible.

---

## Done (historical)

- PR **#35**: consolidated dependency refresh (training + serving).
- **Python 3.13** then **3.14** for root + `model-serving` + CI + Docker + devcontainer; **`model-sim` stays on 3.13 for locks/CI** until TensorFlow supports **3.14** (see row **0**).

When you finish a row in **Own-branch items**, add `Done in PR #___` under it or remove the row if you track issues only in GitHub.
