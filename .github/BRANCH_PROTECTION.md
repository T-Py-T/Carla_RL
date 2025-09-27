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

# Check code style (uses pyproject.toml config)
ruff check .

# Show what would be fixed
ruff check . --diff
```

## Auto-fix Commands

If you want to fix issues locally before pushing:

```bash
# Fix all auto-fixable issues
ruff check --fix .

# Fix only specific rule types
ruff check --fix --select=E,W,F .
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
