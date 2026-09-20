#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/docker"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required to launch CARLA. Install Docker and retry."
  exit 1
fi

echo "Starting CARLA 0.9.15 (requires NVIDIA GPU + nvidia-container-toolkit)..."
docker compose -f docker-compose.carla.yml up -d
echo "CARLA RPC should be available at localhost:2000"
echo "Install the matching Python API egg, then run:"
echo "  cd model-sim && uv run python demos/fsd_playback.py --mode carla --town Town03"
