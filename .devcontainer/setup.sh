 #!/bin/bash
set -e

# Ensure uv is available, install if missing
if ! command -v uv &> /dev/null; then
    echo "[INFO] 'uv' not found. Installing with pip..."
    pip install --upgrade pip
    pip install uv
fi

# Sync dependencies using uv
if [ -f uv.lock ]; then
    echo "[INFO] Installing dependencies from uv.lock..."
    uv sync
else
    echo "[INFO] Installing dependencies from pyproject.toml..."
    uv pip install -e .
fi

echo "[INFO] Carla_RL environment is ready!"
echo "[INFO] You can now run your Python scripts or tests inside the container."
