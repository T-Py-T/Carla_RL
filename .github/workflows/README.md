# GitHub Actions Workflows

This directory contains GitHub Actions workflows for automated CI/CD processes, organized by purpose and trigger type.

## Workflow Categories

### PR-Specific Workflows

#### pr-checks.yml
**Purpose:** Basic quality checks for pull requests
**Triggers:** Pull request opened, synchronized, or reopened
**Features:**
- Python syntax validation
- Basic ruff linting (E9, F rules only)
- Fast execution for quick feedback
- Minimal resource usage

### Merge-Specific Workflows

#### merge-validation.yml
**Purpose:** full validation for merges
**Triggers:** Push to main, master, dev, staging, production branches
**Features:**
- Python syntax validation
- full test suite execution
- Docker build validation
- Configuration file validation
- Full quality assurance

#### feature-release-artifact.yml
**Purpose:** Automated feature release artifact creation
**Triggers:** Pull request closed (merged)
**Features:**
- Extracts PR information and branch details
- Creates structured release artifacts with F001-feature-name naming
- Generates feature summaries and metadata
- Updates release changelog
- Creates GitHub releases
- Cleans up working directories

### ML Pipeline Workflows

#### ml-pipeline-trigger.yml
**Purpose:** Triggers ML pipelines based on code changes
**Triggers:** 
- Push to main/master/dev with changes to model-sim/, training/, models/
- Manual workflow dispatch
**Features:**
- Automatic pipeline type detection (training, evaluation, deployment, full-pipeline)
- Path-based change detection
- Configurable pipeline execution
- Ready for future ML pipeline integration

### Security Workflows

#### security-scan.yml
**Purpose:** Security scanning and vulnerability detection
**Triggers:**
- Daily schedule (2 AM UTC)
- Push to main/master/dev
- Pull requests
- Manual dispatch
**Features:**
- Dependency vulnerability scanning
- Secret detection (with false positive filtering)
- Security report generation
- Artifact upload for reports

## Workflow Design Principles

1. **Separation of Concerns:** Different workflows for different purposes
2. **Performance Optimization:** PR checks are fast, merge validation is full
3. **Future-Ready:** ML pipeline workflows prepared for advanced ML operations
4. **Security-First:** Dedicated security scanning with regular schedules
5. **Maintainability:** Clear naming and purpose for each workflow

## Benefits

- **Automated documentation** of feature releases
- **Structured artifact management** with professional naming
- **full quality assurance** at appropriate stages
- **Security monitoring** with regular scans
- **ML pipeline readiness** for future advanced operations
- **Clean repository maintenance** with automated cleanup

## Usage

### For PRs
- Basic quality checks run automatically
- Fast feedback for developers
- Minimal resource usage

### For Merges
- full validation runs
- Full test suite execution
- Docker build validation
- Feature artifacts created automatically

## Running workflows locally with `act`

Use [nektos/act](https://github.com/nektos/act) with any **Docker-compatible** runtime (**OrbStack**, Docker Desktop, Colima, etc.). Act runs jobs inside containers and approximates GitHub Actions on your machine.

### Setup

1. Install **act** (e.g. `brew install act`).
2. Start **OrbStack** (or your engine) so `docker info` works.
3. Repo root **`.actrc`** sets **`--container-architecture linux/arm64`** for **Apple Silicon + OrbStack**. On **Intel Mac**, edit `.actrc` and use **`linux/amd64`** (see comments in **`.actrc.example`**).

### Commands (Makefile)

| Target | What it runs |
|--------|----------------|
| `make act-list` | Lists jobs act can run |
| `make act-pr` | **pr-checks.yml** — syntax + ruff (fast) |
| `make act-merge` | **merge-validation.yml** — sync, tests, **Docker build** (`--bind` mounts the host Docker socket) |

Workflows are triggered as **`pull_request`** with a small payload in **`.github/act/event-pull-request.json`** so steps that echo PR metadata get sensible values.

### Notes

- **`make act-merge`** needs **`--bind`** so the job can run `docker build` using **your** OrbStack/Docker daemon.
- **Architecture:** `.actrc` defaults to **`linux/arm64`** (Apple Silicon). For **Intel Mac** or strict parity with **GitHub `ubuntu-latest` amd64**, switch to **`linux/amd64`** in `.actrc`.
- Act is **not** identical to GitHub-hosted runners; treat failures as “fix locally or in CI,” not proof the cloud workflow is wrong.

### For ML Operations
- Automatic pipeline detection based on file changes
- Manual trigger for specific pipeline types
- Ready for future ML workflow integration

## Configuration

All workflows use standard GitHub tokens and require no additional setup. They are designed to:
- Run automatically based on appropriate triggers
- Provide clear feedback and reporting
- Maintain security and quality standards
- Support future ML pipeline development