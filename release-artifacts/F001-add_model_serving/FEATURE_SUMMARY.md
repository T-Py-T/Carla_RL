# Feature Release: Add model serving

**Source Branch:** `add_model_serving`  
**Target Branch:** `dev`  
**PR:** #2  
**Merged:** 2025-09-27 18:12:52 UTC  
**Author:** T-Py-T

## Description
<!-- CURSOR_SUMMARY -->
> [!NOTE]
> Rebrands the project to , adds strict Ruff/linting and test tooling, introduces serving PRD/task docs, removes legacy CARLA remnants, and refreshes dependencies/lockfile for Python 3.11+.
> 
> - **Project**:
>   - Rename  → ; update description and  to  in .
> - **Tooling/Config**:
>   - Add comprehensive Ruff config () with linting rules, isort, per-file ignores; dev extras include , , .
> - **Docs/Planning**:
>   - Add serving PRD and task breakdown (, ).
> - **Cleanup**:
>   - Remove legacy CARLA files () and obsolete test compatibility shims under .
> - **Dependencies**:
>   - Overhaul : drop legacy ML stacks, add/upgrade core tooling (, , , ), and align with Python 3.11+.
> 
> <sup>Written by [Cursor Bugbot](https://cursor.com/dashboard?tab=bugbot) for commit 7e60c865ef38a413313cd0ca77987539699958ba. This will update automatically on new commits. Configure [here](https://cursor.com/dashboard?tab=bugbot).</sup>
<!-- /CURSOR_SUMMARY -->

## Files Changed
```
.cursor/rules/create-prd.mdc
.cursor/rules/generate-tasks.mdc
.cursor/rules/process-task-list.mdc
.cursor/rules/release-workflow.mdc
.github/BRANCH_PROTECTION.md
.github/workflows/README.md
.github/workflows/feature-release-artifact.yml
.github/workflows/merge-validation.yml
.github/workflows/ml-pipeline-trigger.yml
.github/workflows/pr-checks.yml
.github/workflows/security-scan.yml
.pre-commit-config.yaml
Makefile
README.md
model-serving/Dockerfile
model-serving/LICENSE
model-serving/Makefile
model-serving/README.md
model-serving/artifacts/v0.1.0/model.pt
model-serving/artifacts/v0.1.0/model_card.yaml
model-serving/artifacts/v0.1.0/preprocessor.pkl
model-serving/create_simple_artifacts.py
model-serving/deploy/README.md
model-serving/deploy/docker/docker-compose.test.yml
model-serving/deploy/docker/load_test.sh
model-serving/deploy/k8s/deployment.yaml
model-serving/deploy/scripts/test_docker_compose.sh
model-serving/docker-compose.yml
model-serving/pyproject.toml
model-serving/requirements.txt
model-serving/scripts/cluster_validation.py
model-serving/scripts/create_example_artifacts.py
model-serving/scripts/test_cluster.sh
model-serving/src/exceptions.py
model-serving/src/inference.py
model-serving/src/io_schemas.py
model-serving/src/model_loader.py
model-serving/src/preprocessing.py
model-serving/src/server.py
model-serving/src/version.py
model-serving/tests/test_api_endpoints.py
model-serving/tests/test_exceptions.py
model-serving/tests/test_inference.py
model-serving/tests/test_inference_qa.py
model-serving/tests/test_integration.py
model-serving/tests/test_model_loader.py
model-serving/tests/test_model_management_qa.py
model-serving/tests/test_preprocessing.py
model-serving/tests/test_qa_plan.py
model-serving/tests/test_schemas.py
model-serving/uv.lock
model-sim/Makefile
model-sim/README.md
model-sim/evaluation/highway/evaluate_models.py
model-sim/pyproject.toml
model-sim/scripts/benchmark.py
model-sim/scripts/control.py
model-sim/scripts/list_gpus.py
model-sim/scripts/play.py
model-sim/scripts/setup_platform.py
model-sim/scripts/train.py
model-sim/src/highway_rl/__init__.py
model-sim/src/highway_rl/agent.py
model-sim/src/highway_rl/environment.py
model-sim/src/highway_rl/logger.py
model-sim/src/highway_rl/trainer.py
model-sim/tests/benchmarks/benchmark_modern_stack.py
model-sim/tests/evaluation/performance_comparison.py
model-sim/tests/evaluation/test_cross_platform.py
model-sim/tests/test_cross_platform.py
model-sim/tests/validation/validate_setup.py
model-sim/training/highway/train_highway.py
model-sim/uv.lock
pyproject.toml
src/carla_mock.py
src/legacy_carla/README.md
src/legacy_carla/carla_rl/__init__.py
src/legacy_carla/carla_rl/agent.py
src/legacy_carla/carla_rl/carla.py
src/legacy_carla/carla_rl/commands.py
src/legacy_carla/carla_rl/common.py
src/legacy_carla/carla_rl/console.py
src/legacy_carla/carla_rl/curriculum_learning.py
src/legacy_carla/carla_rl/models.py
src/legacy_carla/carla_rl/rainbow_agent.py
src/legacy_carla/carla_rl/tensorboard.py
src/legacy_carla/carla_rl/trainer.py
tasks/prd-carla-rl-serving.md
tasks/tasks-prd-carla-rl-serving.md
tests/validation/settings.py
tests/validation/sources.py
uv.lock
```

## Commits in this Feature
```
7e60c86 Fix deprecated GitHub Actions: Update upload-artifact to v4 and setup-uv to v4
74370e7 updates to readme and other scripting files.
8c70bc0 Remove all emojis from workflow files and add strict no-emoji rule
de809c8 Split CI workflows into specialized files
34d5bd9 Improve device detection in PolicyWrapper
8d308a1 Simplify CI checks to basic linting only
45caa4f updates to the linting and formatting of the project environment
6d6f8d6 feat: switch to uv for dependency management
df9cc14 feat: add simple Makefile targets for local linting
44fce95 feat: switch to ruff-only linting for more friendly code checks
967c2ac refactor: improve CI checks and remove local Makefile complexity
d769848 feat: add basic CI checks for PRs
3fc57dc feat: enable release artifacts on merge to any branch
8cbe004 fix: remove manually created release artifacts
1490652 feat: move tasks folder cleanup to final step
97c94c7 fix: clarify tasks/ folder workflow and restore working directory
329fdca docs: update Cursor rules for tasks/ folder workflow
f53c8fa feat: create F001-add-model-serving release artifact (manual test)
ef7493e feat: implement structured feature release naming with F001-feature-name format
74eaa97 refactor: remove emojis from GitHub Action for professional appearance
fbd0e10 feat: add GitHub Action for automated feature release artifacts
351c2cb cleanup: remove legacy CARLA references and focus on Highway RL
ed660c5 refactor: reorganize into function-driven structure
6acd514 fix: add imagePullPolicy for local Kubernetes deployment
ac76f43 refactor: complete rename from carla-rl-serving to model-serving - removed old directory
e4ebf0c refactor: rename service from carla-rl-serving to model-serving
05c417c refactor: organize deployment structure and simplify testing
a899630 feat: complete Infrastructure & Artifact Management for CarlaRL Policy-as-a-Service
4702a6c feat: implement Inference Engine Layer for CarlaRL Policy-as-a-Service
3fe2413 feat: implement Model Management Layer for CarlaRL Policy-as-a-Service
1306516 feat: complete API Layer implementation for CarlaRL Policy-as-a-Service
```
