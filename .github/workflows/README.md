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