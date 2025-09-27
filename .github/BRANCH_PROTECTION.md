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

The project uses `uv` for dependency management. Run checks locally:

```bash
# Install dependencies (if not already done)
uv sync --dev

# Run all quality checks
make check

# Just run linting
make lint

# Auto-fix issues
make fix

# Show what would be fixed
make diff
```

## Manual Commands

If you prefer to run commands directly:

```bash
# Check Python syntax
find . -name "*.py" -exec python -m py_compile {} \;

# Check code style (uses pyproject.toml config)
uv run ruff check .

# Show what would be fixed
uv run ruff check --diff .

# Fix all auto-fixable issues
uv run ruff check --fix .
```

## Ruff Configuration

The project uses a `pyproject.toml` file to configure ruff with:
- **Lenient rules** - Only catches real issues, not style preferences
- **Ignores common false positives** - E501 (line length), W503 (line breaks)
- **Auto-fixable** - Most issues can be automatically fixed
- **Import organization** - Handles import sorting automatically

## Why This Approach?

- **CI Runner**: Has all tools properly installed and configured
- **Consistent Environment**: Same Python version, same tool versions
- **No Local Setup**: Developers don't need to install linting tools
- **Required Checks**: Prevents merging broken code
- **Fast Feedback**: Basic checks run in ~1-2 minutes
