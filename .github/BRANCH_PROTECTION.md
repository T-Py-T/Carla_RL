# Branch Protection Configuration

## Required Status Checks

To ensure code quality, the following CI checks must pass before PRs can be merged:

### Required Checks:
- **basic-checks** - Python syntax, formatting, linting, security, config validation
- **build-and-test** - Dependency installation, tests, Docker build

## GitHub Repository Settings

To enable branch protection:

1. Go to **Settings** → **Branches**
2. Add rule for `main`, `master`, `dev` branches
3. Enable **Require status checks to pass before merging**
4. Select the required checks:
   - `basic-checks`
   - `build-and-test`
5. Enable **Require branches to be up to date before merging**
6. Enable **Restrict pushes that create files larger than 100 MB**

## Local Development

While the CI runner handles all the tooling, you can still run basic checks locally:

```bash
# Check Python syntax
find . -name "*.py" -exec python -m py_compile {} \;

# Check formatting (will show what needs fixing)
black --check --diff .

# Check imports
isort --check-only --diff .

# Basic linting
ruff check . --select=E,W,F
```

## Auto-fix Commands

If you want to fix issues locally before pushing:

```bash
# Fix formatting
black .

# Fix imports  
isort .

# Fix linting issues
ruff check --fix .
```

## Why This Approach?

- **CI Runner**: Has all tools properly installed and configured
- **Consistent Environment**: Same Python version, same tool versions
- **No Local Setup**: Developers don't need to install linting tools
- **Required Checks**: Prevents merging broken code
- **Fast Feedback**: Basic checks run in ~1-2 minutes
