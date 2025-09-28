# Highway RL - Release Changelog

This file tracks all feature releases and their associated artifacts.

## Release History

### F003 - [Staging to Release](https://github.com/T-Py-T/Carla_RL/pull/7) - 2025-09-27 21:22:58 UTC

**Release ID:** `F003-staging`  
**Branch:** `staging`  
**Author:** T-Py-T  
**Artifact:** `release-artifacts/F003-staging/`

* Add model serving (#2)

* feat: complete API Layer implementation for CarlaRL Policy-as-a-Service

- Implemented comprehensive Pydantic schemas with validation

- Created FastAPI server with all required endpoints (/healthz, /metadata, /predict, /warmup)

- Added custom exception handling and error response models

- Built complete test suite with unit tests, integration tests, and QA validation plan

- Includes CORS middleware, request ID tracking, and OpenAPI documentation

- Covers all FR-1.x requirements from PRD with proper error handling

Related to Task 1.0 in PRD-based implementation plan

* feat: implement Model Management Layer for CarlaRL Policy-as-a-Service

- Added PolicyWrapper class with deterministic/stochastic inference modes

- Implemented model loading utilities supporting PyTorch and TorchScript formats

- Created artifact integrity validation with hash pinning and checksums

- Built comprehensive preprocessing pipeline with train-serve parity validation

- Added multi-version model support with semantic versioning

- Implemented device selection logic with CPU/GPU automatic fallback

- Created model metadata parsing from model_card.yaml files

- Added graceful error handling for missing/corrupted artifacts

- Comprehensive test suite with QA validation covering all FR-2.x requirements

Related to Task 2.0 in PRD-based implementation plan

* feat: implement Inference Engine Layer for CarlaRL Policy-as-a-Service

- Created high-performance InferenceEngine with batch processing and memory optimization

- Implemented tensor pre-allocation and memory pinning for reduced latency

- Added torch.no_grad() context and JIT optimization for inference acceleration

- Built deterministic inference mode with reproducible outputs

- Created comprehensive performance timing and metrics collection system

- Implemented batch size optimization and dynamic batching capabilities

- Added version management with git SHA tracking for consistency

- Built inference result caching system for identical inputs optimization

- Implemented graceful degradation and error recovery mechanisms

- Comprehensive test suite with performance benchmarking and QA validation

Related to Task 3.0 in PRD-based implementation plan

* feat: complete Infrastructure & Artifact Management for CarlaRL Policy-as-a-Service

Infrastructure & Deployment (Agent 4):

- Created multi-stage Dockerfile with optimized Python base image and security

- Configured non-root user, read-only filesystem, and resource limits

- Set up comprehensive pyproject.toml with pinned dependencies and dev tools

- Built feature-rich Makefile with development, testing, and deployment commands

- Configured production uvicorn server with worker management

- Implemented environment variable configuration for flexible deployment

- Added Docker health checks and container orchestration support

- Created docker-compose.yml with monitoring stack (Prometheus/Grafana)

- Wrote comprehensive README with setup, usage, and deployment guides

- Built integration tests for containerized deployment validation

Artifact Management System (Agent 5):

- Created model export utilities with TorchScript serialization

- Implemented preprocessor serialization maintaining train-serve parity

- Designed comprehensive model_card.yaml schema with metadata and metrics

- Built semantic versioning artifact directory structure

- Implemented hash-based artifact validation and integrity checking

- Created model registry interface for artifact discovery

- Added example artifacts (v0.1.0) with sample model and preprocessor

- Built validation tests for artifact format compliance

Related to Tasks 4.0 and 5.0 in PRD-based implementation plan

* refactor: organize deployment structure and simplify testing

- Organized deployment files into deploy/docker/ and deploy/k8s/ folders
- Updated Makefile with simple testing commands using existing infrastructure
- Added test-docker, test-compose, and test-k8s targets for easy validation
- Removed overcomplicated bash scripts in favor of Makefile approach
- Clean separation of deployment methods for different environments

* refactor: rename service from carla-rl-serving to model-serving

- Updated all references from CarlaRL to generic model serving
- Changed Docker image names and container references
- Updated Kubernetes deployment manifests
- Modified service names and configuration
- Generalized for Highway RL and other model types
- Maintained all functionality while making it model-agnostic

* refactor: complete rename from carla-rl-serving to model-serving - removed old directory

* fix: add imagePullPolicy for local Kubernetes deployment

- Added imagePullPolicy: Never to use local Docker images
- Enables deployment to local OrbStack Kubernetes cluster
- Tested and validated 3-pod deployment with service access

* refactor: reorganize into function-driven structure

🏗️ Major Repository Reorganization:
- Created model-sim/ for all simulation and training components
- Moved src/, scripts/, training/, evaluation/, tests/ to model-sim/
- Moved pyproject.toml, uv.lock, and simulation Makefile to model-sim/
- Created new root Makefile for component management
- Clean root structure: model-serving/, model-sim/, tasks/, README.md

✨ Benefits:
- Function-driven organization (simulation vs serving vs management)
- Clean root directory with only 4 main components
- Self-contained modules with their own dependencies
- Preserved all existing functionality with updated paths
- Easy navigation and maintenance

* cleanup: remove legacy CARLA references and focus on Highway RL

🧹 Removed Legacy Components:
- Deleted model-sim/src/legacy_carla/ (entire directory)
- Removed model-sim/src/carla_mock.py
- Cleaned up CARLA-dependent test files
- Updated project name from 'carla-rl' to 'highway-rl'

📝 Updated Documentation:
- Changed project descriptions to focus on Highway environments
- Updated Makefile headers and help text
- Removed CARLA references from comments and descriptions

✨ Result: Clean, focused Highway RL codebase

* feat: add GitHub Action for automated feature release artifacts

🤖 Automated Release Management:
- Triggers on PR merge to main/master
- Creates release-artifacts/ folder with feature documentation
- Copies tasks/ folder contents (PRDs, task lists)
- Generates FEATURE_SUMMARY.md with PR details and commits
- Updates RELEASE_CHANGELOG.md with chronological feature history
- Creates GitHub releases with downloadable artifacts
- Fully automated - no manual intervention required

✨ Benefits:
- Preserves feature planning and development history
- Enables project retrospectives and learning
- Professional release management
- Maintains clean project documentation
- Supports compliance and audit requirements

* refactor: remove emojis from GitHub Action for professional appearance

- Cleaned up all emoji usage in workflow output messages
- Updated README documentation to be emoji-free
- Maintained all functionality while improving professional appearance
- Action output now uses clean, readable text formatting

* feat: implement structured feature release naming with F001-feature-name format

- Added automatic feature numbering (F001, F002, etc.)
- Sanitized branch names for clean folder structure
- Updated all references to use new naming convention
- Enhanced metadata with feature release ID and number
- Improved changelog format with structured release IDs
- GitHub releases now use F-number format for better organization

Example: F001-add-model-serving for add_model_serving branch

* feat: create F001-add-model-serving release artifact (manual test)

This is the inaugural structured feature release demonstrating the new release artifact system. Created manually to test and validate the workflow before automating via GitHub Actions.

Release Contents:
- F001-add-model-serving/ folder with complete documentation
- PRD and task breakdown preserved from development
- Auto-generated FEATURE_SUMMARY.md with comprehensive details
- Structured metadata.json for machine processing
- RELEASE_CHANGELOG.md tracking project history

This release represents the complete Model Serving microservice implementation with FastAPI, Docker, Kubernetes, and comprehensive testing capabilities.

Next: GitHub Action will automatically create F002, F003, etc.

* docs: update Cursor rules for tasks/ folder workflow

Updated Cursor MDC rules to reflect established patterns:
- PRDs and task lists always saved to tasks/ folder
- Documents automatically preserved in release artifacts
- Added new release-workflow.mdc with complete guidelines
- Updated create-prd.mdc and generate-tasks.mdc references
- Ensures consistency with F001-feature-name workflow

This aligns AI assistant behavior with our structured release artifact system and GitHub Actions automation.

* fix: clarify tasks/ folder workflow and restore working directory

WORKFLOW CLARIFICATION:
1. During development: Work in tasks/ folder (PRDs, task lists)
2. On merge: GitHub Action copies tasks/ to release-artifacts/F001-feature-name/
3. After artifact creation: Clean up tasks/ folder for next feature

CHANGES:
- Restored working tasks/ folder with current feature docs
- Updated GitHub Action to clean up tasks/ after artifact creation
- Added git add -A to properly stage folder deletions
- Updated commit message to reflect cleanup step

This ensures clean separation between working docs and preserved artifacts.

* feat: move tasks folder cleanup to final step

WORKFLOW IMPROVEMENT:
- Moved tasks/ folder cleanup to be the very last step
- Cleanup now happens AFTER all artifacts are created and committed
- Ensures complete preservation before any cleanup occurs
- Final step: 'Final cleanup - Remove tasks folder'

SEQUENCE:
1. Create release artifact from tasks/ content
2. Generate summaries and metadata
3. Update changelog
4. Commit all changes
5. Create GitHub release
6. Upload release assets
7. Show summary
8. FINAL: Remove tasks/ folder (ready for next feature)

This guarantees no data loss and clean workspace for next feature.

* fix: remove manually created release artifacts

CORRECTION: I incorrectly created release artifacts manually when these should ONLY be created by the GitHub Action on merge completion.

REMOVED:
- release-artifacts/ folder (should be created by GitHub Action)
- RELEASE_CHANGELOG.md (should be created by GitHub Action)

CORRECT WORKFLOW:
1. Work in tasks/ folder during development
2. Merge PR triggers GitHub Action
3. GitHub Action creates release-artifacts/F001-feature-name/
4. GitHub Action creates RELEASE_CHANGELOG.md
5. GitHub Action cleans up tasks/ folder

The branch should only contain:
- tasks/ folder with working docs
- GitHub Action workflow
- Model serving implementation
- No pre-created release artifacts

* feat: enable release artifacts on merge to any branch

ENHANCEMENTS:
- Triggers on merge to ANY branch (not just main/master)
- Added target branch information to release metadata
- Enhanced feature summary with source/target branch details
- Updated metadata.json with target_branch field

WORKFLOW BENEFITS:
- Merge to dev → creates release artifact
- Merge to staging → creates release artifact
- Merge to production → creates release artifact
- Full control over when releases are created
- Clear tracking of which environment triggered the release

This supports proper staging environments where you merge feature branches to dev first, then promote to production.

* feat: add basic CI checks for PRs

CI WORKFLOW:
- Step 1: Basic sanity checks (syntax, format, lint, security)
- Step 2: Build and test (only if basic checks pass)
- Step 3: Summary with clear pass/fail status

BASIC CHECKS (fail fast):
- Python syntax validation
- Code formatting (black)
- Import sorting (isort)
- Basic linting (ruff)
- Security scan (hardcoded secrets)
- Config file validation (YAML/JSON)

BUILD & TEST:
- Install dependencies
- Run pytest tests
- Test Docker build

LOCAL COMMANDS:
- make lint - Run linting checks
- make format - Auto-fix formatting
- make test - Run tests
- make ci-checks - Run all checks locally

This provides fast feedback on basic issues before running expensive build and test operations.

* refactor: improve CI checks and remove local Makefile complexity

IMPROVEMENTS:
- Pinned tool versions for consistency (ruff==0.1.8, black==23.12.1, isort==5.13.2)
- Better error handling with proper exit codes
- Excluded .venv and .git directories from checks
- Improved security scan to exclude examples/tests/placeholders
- Enhanced config file validation with proper error messages
- Removed complex Makefile targets (CI runner handles tooling)

CI WORKFLOW:
- Step 1: Basic checks (syntax, format, lint, security, config)
- Step 2: Build & test (only if basic checks pass)
- Step 3: Clear summary with pass/fail status

BRANCH PROTECTION:
- Added BRANCH_PROTECTION.md with setup instructions
- CI checks will be required for PR approval
- Fast feedback in 1-2 minutes

This approach lets the CI runner handle all tooling while providing fast, reliable feedback on PRs.

* feat: switch to ruff-only linting for more friendly code checks

IMPROVEMENTS:
- Removed black and isort (too opinionated)
- Using ruff for both linting and formatting
- Added pyproject.toml with lenient configuration
- Ignores common false positives (E501, W503)
- Auto-fixable rules for easy local fixes

RUFF CONFIGURATION:
- Only catches real issues, not style preferences
- Ignores line length and line break warnings
- Handles import organization automatically
- Most issues can be auto-fixed with 'ruff check --fix .'

LOCAL USAGE:
- 'ruff check .' - Check for issues
- 'ruff check --diff .' - Show what would be fixed
- 'ruff check --fix .' - Auto-fix issues

This approach is much more forgiving and focuses on real code quality issues rather than style preferences.

* feat: add simple Makefile targets for local linting

NEW TARGETS:
- make check    - Run all quality checks (syntax + linting)
- make lint     - Run ruff linting only
- make fix      - Auto-fix code issues
- make format   - Auto-format code (same as fix)
- make diff     - Show what would be fixed

FEATURES:
- Auto-installs ruff if not present
- Clear success/failure messages
- Excludes .venv and .git directories
- Uses pyproject.toml configuration
- Helpful error messages with fix suggestions

USAGE:
- 'make check'  - Run before committing
- 'make fix'    - Auto-fix issues
- 'make diff'   - Preview changes

This provides a simple way to run the same checks
locally that the CI runner will perform.

* feat: switch to uv for dependency management

MAJOR CHANGES:
- Updated Makefile to use 'uv run' instead of pip
- Added ruff to pyproject.toml dependencies
- Updated CI workflow to use uv instead of pip
- Added astral-sh/setup-uv action for CI

MAKEFILE IMPROVEMENTS:
- 'make check' now uses 'uv run ruff check .'
- 'make fix' uses 'uv run ruff check --fix .'
- All commands work with existing .venv
- No more pip installation attempts

CI WORKFLOW:
- Uses 'uv sync --dev' to install dependencies
- Uses 'uv run' for all Python commands
- Consistent with local development workflow

PYPROJECT.TOML:
- Added proper project metadata
- ruff>=0.1.8 in dependencies
- pytest and pytest-cov in dev dependencies
- Maintains existing ruff configuration

This ensures consistent dependency management
between local development and CI environment.

* updates to the linting and formatting of the project environment

* Simplify CI checks to basic linting only

- Remove complex multi-stage CI workflow
- Keep only essential checks: Python syntax + minimal ruff (E9,F rules)
- Exclude virtual environments and build directories from syntax check
- Use minimal ruff rules to avoid conflicts with local development
- Local development remains slightly stricter than CI pipeline

* Improve device detection in PolicyWrapper

- Add safer device detection with proper error handling
- Handle cases where model has no parameters
- Add fallback to CPU device if detection fails
- Prevent StopIteration and AttributeError exceptions

* Split CI workflows into specialized files

- pr-checks.yml: Fast quality checks for PRs (syntax + basic linting)
- merge-validation.yml: Comprehensive validation for merges (tests + Docker + config)
- ml-pipeline-trigger.yml: ML pipeline automation (training/eval/deployment)
- security-scan.yml: Security scanning and vulnerability detection
- Updated README.md with comprehensive workflow documentation

Benefits:
- Separation of concerns for different workflow purposes
- Performance optimization (fast PR checks, comprehensive merge validation)
- Future-ready ML pipeline integration
- Security-first approach with regular scans
- Clear naming and maintainable structure

* Remove all emojis from workflow files and add strict no-emoji rule

- Removed all emojis from pr-checks.yml, merge-validation.yml, ml-pipeline-trigger.yml, security-scan.yml
- Created .cursor/rules/no-emojis.mdc with mandatory no-emoji policy
- All echo statements now use plain text
- Professional appearance maintained throughout

This ensures all future files will be emoji-free as requested.

* updates to readme and other scripting files.

* Fix deprecated GitHub Actions: Update upload-artifact to v4 and setup-uv to v4

- Updated actions/upload-artifact from v3 to v4 in security-scan.yml
- Updated astral-sh/setup-uv from v3 to v4 in all workflows
- Fixes security scan failure due to deprecated action versions

* feat: add release artifact F001 for Add model serving

  - Created artifact: release-artifacts/F001-add_model_serving/
  - Updated RELEASE_CHANGELOG.md with feature summary
  - Preserved tasks folder contents and metadata
  - Tasks folder will be cleaned up in final step

  Release ID: F001-add_model_serving
  PR: #2
  Branch: add_model_serving

* Staging (#4)

* Dev (#3)

* Add model serving (#2)

* feat: complete API Layer implementation for CarlaRL Policy-as-a-Service

- Implemented comprehensive Pydantic schemas with validation

- Created FastAPI server with all required endpoints (/healthz, /metadata, /predict, /warmup)

- Added custom exception handling and error response models

- Built complete test suite with unit tests, integration tests, and QA validation plan

- Includes CORS middleware, request ID tracking, and OpenAPI documentation

- Covers all FR-1.x requirements from PRD with proper error handling

Related to Task 1.0 in PRD-based implementation plan

* feat: implement Model Management Layer for CarlaRL Policy-as-a-Service

- Added PolicyWrapper class with deterministic/stochastic inference modes

- Implemented model loading utilities supporting PyTorch and TorchScript formats

- Created artifact integrity validation with hash pinning and checksums

- Built comprehensive preprocessing pipeline with train-serve parity validation

- Added multi-version model support with semantic versioning

- Implemented device selection logic with CPU/GPU automatic fallback

- Created model metadata parsing from model_card.yaml files

- Added graceful error handling for missing/corrupted artifacts

- Comprehensive test suite with QA validation covering all FR-2.x requirements

Related to Task 2.0 in PRD-based implementation plan

* feat: implement Inference Engine Layer for CarlaRL Policy-as-a-Service

- Created high-performance InferenceEngine with batch processing and memory optimization

- Implemented tensor pre-allocation and memory pinning for reduced latency

- Added torch.no_grad() context and JIT optimization for inference acceleration

- Built deterministic inference mode with reproducible outputs

- Created comprehensive performance timing and metrics collection system

- Implemented batch size optimization and dynamic batching capabilities

- Added version management with git SHA tracking for consistency

- Built inference result caching system for identical inputs optimization

- Implemented graceful degradation and error recovery mechanisms

- Comprehensive test suite with performance benchmarking and QA validation

Related to Task 3.0 in PRD-based implementation plan

* feat: complete Infrastructure & Artifact Management for CarlaRL Policy-as-a-Service

Infrastructure & Deployment (Agent 4):

- Created multi-stage Dockerfile with optimized Python base image and security

- Configured non-root user, read-only filesystem, and resource limits

- Set up comprehensive pyproject.toml with pinned dependencies and dev tools

- Built feature-rich Makefile with development, testing, and deployment commands

- Configured production uvicorn server with worker management

- Implemented environment variable configuration for flexible deployment

- Added Docker health checks and container orchestration support

- Created docker-compose.yml with monitoring stack (Prometheus/Grafana)

- Wrote comprehensive README with setup, usage, and deployment guides

- Built integration tests for containerized deployment validation

Artifact Management System (Agent 5):

- Created model export utilities with TorchScript serialization

- Implemented preprocessor serialization maintaining train-serve parity

- Designed comprehensive model_card.yaml schema with metadata and metrics

- Built semantic versioning artifact directory structure

- Implemented hash-based artifact validation and integrity checking

- Created model registry interface for artifact discovery

- Added example artifacts (v0.1.0) with sample model and preprocessor

- Built validation tests for artifact format compliance

Related to Tasks 4.0 and 5.0 in PRD-based implementation plan

* refactor: organize deployment structure and simplify testing

- Organized deployment files into deploy/docker/ and deploy/k8s/ folders
- Updated Makefile with simple testing commands using existing infrastructure
- Added test-docker, test-compose, and test-k8s targets for easy validation
- Removed overcomplicated bash scripts in favor of Makefile approach
- Clean separation of deployment methods for different environments

* refactor: rename service from carla-rl-serving to model-serving

- Updated all references from CarlaRL to generic model serving
- Changed Docker image names and container references
- Updated Kubernetes deployment manifests
- Modified service names and configuration
- Generalized for Highway RL and other model types
- Maintained all functionality while making it model-agnostic

* refactor: complete rename from carla-rl-serving to model-serving - removed old directory

* fix: add imagePullPolicy for local Kubernetes deployment

- Added imagePullPolicy: Never to use local Docker images
- Enables deployment to local OrbStack Kubernetes cluster
- Tested and validated 3-pod deployment with service access

* refactor: reorganize into function-driven structure

🏗️ Major Repository Reorganization:
- Created model-sim/ for all simulation and training components
- Moved src/, scripts/, training/, evaluation/, tests/ to model-sim/
- Moved pyproject.toml, uv.lock, and simulation Makefile to model-sim/
- Created new root Makefile for component management
- Clean root structure: model-serving/, model-sim/, tasks/, README.md

✨ Benefits:
- Function-driven organization (simulation vs serving vs management)
- Clean root directory with only 4 main components
- Self-contained modules with their own dependencies
- Preserved all existing functionality with updated paths
- Easy navigation and maintenance

* cleanup: remove legacy CARLA references and focus on Highway RL

🧹 Removed Legacy Components:
- Deleted model-sim/src/legacy_carla/ (entire directory)
- Removed model-sim/src/carla_mock.py
- Cleaned up CARLA-dependent test files
- Updated project name from 'carla-rl' to 'highway-rl'

📝 Updated Documentation:
- Changed project descriptions to focus on Highway environments
- Updated Makefile headers and help text
- Removed CARLA references from comments and descriptions

✨ Result: Clean, focused Highway RL codebase

* feat: add GitHub Action for automated feature release artifacts

🤖 Automated Release Management:
- Triggers on PR merge to main/master
- Creates release-artifacts/ folder with feature documentation
- Copies tasks/ folder contents (PRDs, task lists)
- Generates FEATURE_SUMMARY.md with PR details and commits
- Updates RELEASE_CHANGELOG.md with chronological feature history
- Creates GitHub releases with downloadable artifacts
- Fully automated - no manual intervention required

✨ Benefits:
- Preserves feature planning and development history
- Enables project retrospectives and learning
- Professional release management
- Maintains clean project documentation
- Supports compliance and audit requirements

* refactor: remove emojis from GitHub Action for professional appearance

- Cleaned up all emoji usage in workflow output messages
- Updated README documentation to be emoji-free
- Maintained all functionality while improving professional appearance
- Action output now uses clean, readable text formatting

* feat: implement structured feature release naming with F001-feature-name format

- Added automatic feature numbering (F001, F002, etc.)
- Sanitized branch names for clean folder structure
- Updated all references to use new naming convention
- Enhanced metadata with feature release ID and number
- Improved changelog format with structured release IDs
- GitHub releases now use F-number format for better organization

Example: F001-add-model-serving for add_model_serving branch

* feat: create F001-add-model-serving release artifact (manual test)

This is the inaugural structured feature release demonstrating the new release artifact system. Created manually to test and validate the workflow before automating via GitHub Actions.

Release Contents:
- F001-add-model-serving/ folder with complete documentation
- PRD and task breakdown preserved from development
- Auto-generated FEATURE_SUMMARY.md with comprehensive details
- Structured metadata.json for machine processing
- RELEASE_CHANGELOG.md tracking project history

This release represents the complete Model Serving microservice implementation with FastAPI, Docker, Kubernetes, and comprehensive testing capabilities.

Next: GitHub Action will automatically create F002, F003, etc.

* docs: update Cursor rules for tasks/ folder workflow

Updated Cursor MDC rules to reflect established patterns:
- PRDs and task lists always saved to tasks/ folder
- Documents automatically preserved in release artifacts
- Added new release-workflow.mdc with complete guidelines
- Updated create-prd.mdc and generate-tasks.mdc references
- Ensures consistency with F001-feature-name workflow

This aligns AI assistant behavior with our structured release artifact system and GitHub Actions automation.

* fix: clarify tasks/ folder workflow and restore working directory

WORKFLOW CLARIFICATION:
1. During development: Work in tasks/ folder (PRDs, task lists)
2. On merge: GitHub Action copies tasks/ to release-artifacts/F001-feature-name/
3. After artifact creation: Clean up tasks/ folder for next feature

CHANGES:
- Restored working tasks/ folder with current feature docs
- Updated GitHub Action to clean up tasks/ after artifact creation
- Added git add -A to properly stage folder deletions
- Updated commit message to reflect cleanup step

This ensures clean separation between working docs and preserved artifacts.

* feat: move tasks folder cleanup to final step

WORKFLOW IMPROVEMENT:
- Moved tasks/ folder cleanup to be the very last step
- Cleanup now happens AFTER all artifacts are created and committed
- Ensures complete preservation before any cleanup occurs
- Final step: 'Final cleanup - Remove tasks folder'

SEQUENCE:
1. Create release artifact from tasks/ content
2. Generate summaries and metadata
3. Update changelog
4. Commit all changes
5. Create GitHub release
6. Upload release assets
7. Show summary
8. FINAL: Remove tasks/ folder (ready for next feature)

This guarantees no data loss and clean workspace for next feature.

* fix: remove manually created release artifacts

CORRECTION: I incorrectly created release artifacts manually when these should ONLY be created by the GitHub Action on merge completion.

REMOVED:
- release-artifacts/ folder (should be created by GitHub Action)
- RELEASE_CHANGELOG.md (should be created by GitHub Action)

CORRECT WORKFLOW:
1. Work in tasks/ folder during development
2. Merge PR triggers GitHub Action
3. GitHub Action creates release-artifacts/F001-feature-name/
4. GitHub Action creates RELEASE_CHANGELOG.md
5. GitHub Action cleans up tasks/ folder

The branch should only contain:
- tasks/ folder with working docs
- GitHub Action workflow
- Model serving implementation
- No pre-created release artifacts

* feat: enable release artifacts on merge to any branch

ENHANCEMENTS:
- Triggers on merge to ANY branch (not just main/master)
- Added target branch information to release metadata
- Enhanced feature summary with source/target branch details
- Updated metadata.json with target_branch field

WORKFLOW BENEFITS:
- Merge to dev → creates release artifact
- Merge to staging → creates release artifact
- Merge to production → creates release artifact
- Full control over when releases are created
- Clear tracking of which environment triggered the release

This supports proper staging environments where you merge feature branches to dev first, then promote to production.

* feat: add basic CI checks for PRs

CI WORKFLOW:
- Step 1: Basic sanity checks (syntax, format, lint, security)
- Step 2: Build and test (only if basic checks pass)
- Step 3: Summary with clear pass/fail status

BASIC CHECKS (fail fast):
- Python syntax validation
- Code formatting (black)
- Import sorting (isort)
- Basic linting (ruff)
- Security scan (hardcoded secrets)
- Config file validation (YAML/JSON)

BUILD & TEST:
- Install dependencies
- Run pytest tests
- Test Docker build

LOCAL COMMANDS:
- make lint - Run linting checks
- make format - Auto-fix formatting
- make test - Run tests
- make ci-checks - Run all checks locally

This provides fast feedback on basic issues before running expensive build and test operations.

* refactor: improve CI checks and remove local Makefile complexity

IMPROVEMENTS:
- Pinned tool versions for consistency (ruff==0.1.8, black==23.12.1, isort==5.13.2)
- Better error handling with proper exit codes
- Excluded .venv and .git directories from checks
- Improved security scan to exclude examples/tests/placeholders
- Enhanced config file validation with proper error messages
- Removed complex Makefile targets (CI runner handles tooling)

CI WORKFLOW:
- Step 1: Basic checks (syntax, format, lint, security, config)
- Step 2: Build & test (only if basic checks pass)
- Step 3: Clear summary with pass/fail status

BRANCH PROTECTION:
- Added BRANCH_PROTECTION.md with setup instructions
- CI checks will be required for PR approval
- Fast feedback in 1-2 minutes

This approach lets the CI runner handle all tooling while providing fast, reliable feedback on PRs.

* feat: switch to ruff-only linting for more friendly code checks

IMPROVEMENTS:
- Removed black and isort (too opinionated)
- Using ruff for both linting and formatting
- Added pyproject.toml with lenient configuration
- Ignores common false positives (E501, W503)
- Auto-fixable rules for easy local fixes

RUFF CONFIGURATION:
- Only catches real issues, not style preferences
- Ignores line length and line break warnings
- Handles import organization automatically
- Most issues can be auto-fixed with 'ruff check --fix .'

LOCAL USAGE:
- 'ruff check .' - Check for issues
- 'ruff check --diff .' - Show what would be fixed
- 'ruff check --fix .' - Auto-fix issues

This approach is much more forgiving and focuses on real code quality issues rather than style preferences.

* feat: add simple Makefile targets for local linting

NEW TARGETS:
- make check    - Run all quality checks (syntax + linting)
- make lint     - Run ruff linting only
- make fix      - Auto-fix code issues
- make format   - Auto-format code (same as fix)
- make diff     - Show what would be fixed

FEATURES:
- Auto-installs ruff if not present
- Clear success/failure messages
- Excludes .venv and .git directories
- Uses pyproject.toml configuration
- Helpful error messages with fix suggestions

USAGE:
- 'make check'  - Run before committing
- 'make fix'    - Auto-fix issues
- 'make diff'   - Preview changes

This provides a simple way to run the same checks
locally that the CI runner will perform.

* feat: switch to uv for dependency management

MAJOR CHANGES:
- Updated Makefile to use 'uv run' instead of pip
- Added ruff to pyproject.toml dependencies
- Updated CI workflow to use uv instead of pip
- Added astral-sh/setup-uv action for CI

MAKEFILE IMPROVEMENTS:
- 'make check' now uses 'uv run ruff check .'
- 'make fix' uses 'uv run ruff check --fix .'
- All commands work with existing .venv
- No more pip installation attempts

CI WORKFLOW:
- Uses 'uv sync --dev' to install dependencies
- Uses 'uv run' for all Python commands
- Consistent with local development workflow

PYPROJECT.TOML:
- Added proper project metadata
- ruff>=0.1.8 in dependencies
- pytest and pytest-cov in dev dependencies
- Maintains existing ruff configuration

This ensures consistent dependency management
between local development and CI environment.

* updates to the linting and formatting of the project environment

* Simplify CI checks to basic linting only

- Remove complex multi-stage CI workflow
- Keep only essential checks: Python syntax + minimal ruff (E9,F rules)
- Exclude virtual environments and build directories from syntax check
- Use minimal ruff rules to avoid conflicts with local development
- Local development remains slightly stricter than CI pipeline

* Improve device detection in PolicyWrapper

- Add safer device detection with proper error handling
- Handle cases where model has no parameters
- Add fallback to CPU device if detection fails
- Prevent StopIteration and AttributeError exceptions

* Split CI workflows into specialized files

- pr-checks.yml: Fast quality checks for PRs (syntax + basic linting)
- merge-validation.yml: Comprehensive validation for merges (tests + Docker + config)
- ml-pipeline-trigger.yml: ML pipeline automation (training/eval/deployment)
- security-scan.yml: Security scanning and vulnerability detection
- Updated README.md with comprehensive workflow documentation

Benefits:
- Separation of concerns for different workflow purposes
- Performance optimization (fast PR checks, comprehensive merge validation)
- Future-ready ML pipeline integration
- Security-first approach with regular scans
- Clear naming and maintainable structure

* Remove all emojis from workflow files and add strict no-emoji rule

- Removed all emojis from pr-checks.yml, merge-validation.yml, ml-pipeline-trigger.yml, security-scan.yml
- Created .cursor/rules/no-emojis.mdc with mandatory no-emoji policy
- All echo statements now use plain text
- Professional appearance maintained throughout

This ensures all future files will be emoji-free as requested.

* updates to readme and other scripting files.

* Fix deprecated GitHub Actions: Update upload-artifact to v4 and setup-uv to v4

- Updated actions/upload-artifact from v3 to v4 in security-scan.yml
- Updated astral-sh/setup-uv from v3 to v4 in all workflows
- Fixes security scan failure due to deprecated action versions

* feat: add release artifact F001 for Add model serving

  - Created artifact: release-artifacts/F001-add_model_serving/
  - Updated RELEASE_CHANGELOG.md with feature summary
  - Preserved tasks folder contents and metadata
  - Tasks folder will be cleaned up in final step

  Release ID: F001-add_model_serving
  PR: #2
  Branch: add_model_serving

---------



* feat: add release artifact F002 for Dev

  - Created artifact: release-artifacts/F002-dev/
  - Updated RELEASE_CHANGELOG.md with feature summary
  - Preserved tasks folder contents and metadata
  - Tasks folder will be cleaned up in final step

  Release ID: F002-dev
  PR: #3
  Branch: dev

---------



* feat: add release artifact F003 for Staging

  - Created artifact: release-artifacts/F003-staging/
  - Updated RELEASE_CHANGELOG.md with feature summary
  - Preserved tasks folder contents and metadata
  - Tasks folder will be cleaned up in final step

  Release ID: F003-staging
  PR: #4
  Branch: staging

* Bump torch from 2.1.1 to 2.8.0 in /model-serving (#5)

Bumps [torch](https://github.com/pytorch/pytorch) from 2.1.1 to 2.8.0.
- [Release notes](https://github.com/pytorch/pytorch/releases)
- [Changelog](https://github.com/pytorch/pytorch/blob/main/RELEASE.md)
- [Commits](https://github.com/pytorch/pytorch/compare/v2.1.1...v2.8.0)

---
updated-dependencies:
- dependency-name: torch dependency-version: 2.8.0 dependency-type: direct:production ...




* feat: add release artifact F004 for Bump torch from 2.1.1 to 2.8.0 in /model-serving

  - Created artifact: release-artifacts/F004-dependabot-pip-model-serving-torch-2-8-0/
  - Updated RELEASE_CHANGELOG.md with feature summary
  - Preserved tasks folder contents and metadata
  - Tasks folder will be cleaned up in final step

  Release ID: F004-dependabot-pip-model-serving-torch-2-8-0
  PR: #5
  Branch: dependabot/pip/model-serving/torch-2.8.0

* Fix GitHub Action: Only trigger on release branch and move changelog to release-artifacts/

- Updated workflow to only trigger on merges to 'release' branch (not dev/staging)
- Moved RELEASE_CHANGELOG.md to release-artifacts/ folder to avoid root clutter
- Updated all references to point to release-artifacts/RELEASE_CHANGELOG.md

This prevents creating release artifacts on every merge and keeps all release-related files organized in the release-artifacts/ folder.

* removing artifacts from dev, they should only be in staging.

* Add GitHub Workflow SOP for Cursor agents

- Created comprehensive SOP for GitHub workflow procedures
- Covers branch strategy, development workflow, and release management
- Includes commit standards, file organization rules, and best practices
- Set alwaysApply: true for consistent application across repositories
- Provides clear guidance for Cursor agents on proper GitHub operations

* Fix merge validation workflow pytest dependency issue

- Install dependencies for each component separately (model-serving, model-sim)
- Run pytest from within each component directory to use correct environment
- Add proper error handling and logging for test execution
- Ensures pytest is available in the correct virtual environment for each component

* Restructure CI workflows for proper PR validation

- Move comprehensive tests to PR-triggered merge-validation workflow
- Create lightweight branch-validation for push events
- Add PR information display for better context
- Create documentation for required status checks setup
- Ensures fast feedback on pushes, thorough validation on PRs
- Prevents merge until comprehensive tests pass

* Fix CI workflow performance and Docker warnings

- Replace while loops with find -exec for better performance and no hanging
- Fix Docker FROM statement casing warnings (as -> AS)
- Standardize user/group names to modelserving throughout Dockerfile
- Improve error handling and prevent runaway processes in CI
- Reduce CI execution time and eliminate Docker build warnings

* cleanup: update to layout

---------

**Key Changes:**
- .cursor/rules/github-workflow-sop.mdc
- .github/REQUIRED_STATUS_CHECKS.md
- .github/workflows/branch-validation.yml
- .github/workflows/feature-release-artifact.yml
- .github/workflows/merge-validation.yml
- =3.11
- RELEASE_CHANGELOG.md
- model-serving/Dockerfile


---

