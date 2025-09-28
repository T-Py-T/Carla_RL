#!/bin/bash

# Local PR Quality Checks - mimics GitHub Actions workflow
echo "Running local PR quality checks..."

cd model-serving

echo "Checking Python syntax..."
# Check syntax for all Python files
find . -name "*.py" -exec python3 -m py_compile {} \;
if [ $? -ne 0 ]; then
    echo "Syntax error found!"
    exit 1
fi
echo "Python syntax check passed!"

echo "Running linting with ruff..."
uv run ruff check src/ tests/
if [ $? -ne 0 ]; then
    echo "Linting issues found!"
    exit 1
fi
echo "Linting check passed!"

echo "Running type checking with mypy..."
uv run mypy src/
if [ $? -ne 0 ]; then
    echo "Type checking issues found!"
    exit 1
fi
echo "Type checking passed!"

echo "Running tests..."
uv run pytest tests/ -v
if [ $? -ne 0 ]; then
    echo "Tests failed!"
    exit 1
fi
echo "All tests passed!"

echo "All PR quality checks passed locally!"
