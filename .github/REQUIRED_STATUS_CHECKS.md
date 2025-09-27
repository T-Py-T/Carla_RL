# Required Status Checks Setup

This document explains how to configure the required status checks for proper PR validation workflow.

## Workflow Overview

We have two validation workflows:

1. **Branch Validation** (`branch-validation.yml`)
   - Triggers: Push to protected branches
   - Purpose: Light validation (syntax, basic linting, config validation)
   - Duration: ~30 seconds

2. **Merge Validation** (`merge-validation.yml`)
   - Triggers: PR opened/updated
   - Purpose: Comprehensive validation (full tests, Docker builds)
   - Duration: ~5-10 minutes

## Required Status Checks Configuration

### For each protected branch (main, dev, staging, release):

1. Go to **Settings** → **Branches** → **Branch protection rules**
2. Select the branch (e.g., `main`)
3. Enable **Require status checks to pass before merging**
4. Add these required status checks:
   - `merge-validation` (comprehensive PR validation)
   - `branch-validation` (basic branch validation)

### Recommended Settings:

- ✅ Require status checks to pass before merging
- ✅ Require branches to be up to date before merging
- ✅ Require conversation resolution before merging
- ✅ Require review from code owners
- ✅ Restrict pushes that create files larger than 100 MB

## Workflow Behavior

### On PR Creation/Update:
- `merge-validation` runs (comprehensive tests)
- PR cannot be merged until this passes
- Takes 5-10 minutes

### On Direct Push to Protected Branch:
- `branch-validation` runs (light validation)
- Fails fast if there are syntax errors
- Takes ~30 seconds

## Benefits

1. **Fast Feedback**: Basic issues caught immediately on push
2. **Thorough Validation**: Comprehensive testing before merge
3. **No Blocking**: Developers can push to feature branches without waiting
4. **Quality Gate**: Only tested code reaches protected branches

## Troubleshooting

### If merge-validation is not showing as required:
1. Check that the workflow file is in `.github/workflows/`
2. Verify the workflow has run at least once
3. Ensure the job name matches exactly: `merge-validation`
4. Check branch protection rules are properly configured

### If tests are failing:
1. Check the workflow logs for specific error messages
2. Ensure all dependencies are properly installed
3. Verify test files are in the correct locations
4. Check that pytest is available in the component directories
