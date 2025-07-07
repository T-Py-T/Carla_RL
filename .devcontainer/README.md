 # Carla_RL Dev Container

This dev container provides a reproducible development environment for the Carla_RL project using [uv](https://github.com/astral-sh/uv) for fast Python dependency management.

## Features
- Python 3.11 (slim image)
- [uv](https://github.com/astral-sh/uv) for dependency management
- Dependencies installed from `pyproject.toml` and `uv.lock`
- VS Code Python and Pylance extensions
- Port 8888 forwarded (for Jupyter, etc.)

## Usage
1. Open this folder in [VS Code](https://code.visualstudio.com/) with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).
2. Reopen in Container when prompted.
3. On first start, dependencies will be installed using `uv` from `uv.lock` (if present) or `pyproject.toml`.
4. Start developing!

## Notes
- The workspace is mounted at `/workspace` inside the container.
- To install new dependencies, add them to `pyproject.toml` and run `uv pip install -e .` inside the container.
