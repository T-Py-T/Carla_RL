#!/usr/bin/env bash
# Run GitHub Actions jobs locally via nektos/act (Docker / OrbStack).
# Usage (repo root): ./.github/scripts/act-all.sh
# Optional: ACT=/path/to/act

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

ACT_BIN="${ACT:-$(command -v act || true)}"
if [[ -z "$ACT_BIN" || ! -x "$ACT_BIN" ]]; then
  echo "act not found. Install: https://github.com/nektos/act/releases" >&2
  exit 127
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker is not reachable (start OrbStack / Docker Desktop, or run dockerd)." >&2
  exit 1
fi

ARCH="$(uname -m)"
if [[ "$ARCH" == "arm64" || "$ARCH" == "aarch64" ]]; then
  CONTAINER_ARCH="linux/arm64"
else
  CONTAINER_ARCH="linux/amd64"
fi

EVENT_PR="$ROOT/.github/act/event-pull-request.json"
COMMON=(--container-architecture "$CONTAINER_ARCH")

run() {
  local name="$1"
  shift
  echo ""
  echo "========== $name =========="
  "$ACT_BIN" "$@" "${COMMON[@]}"
}

run "branch-validation (push)" push \
  -W "$ROOT/.github/workflows/branch-validation.yml" \
  -j branch-validation

run "pr-checks (pull_request)" pull_request \
  -W "$ROOT/.github/workflows/pr-checks.yml" \
  -j pr-quality-checks \
  -e "$EVENT_PR"

run "merge-validation (pull_request + docker)" pull_request \
  -W "$ROOT/.github/workflows/merge-validation.yml" \
  -j merge-validation \
  -e "$EVENT_PR" \
  --bind

run "security-scan (workflow_dispatch)" workflow_dispatch \
  -W "$ROOT/.github/workflows/security-scan.yml" \
  -j security-scan

run "ml-pipeline-trigger (workflow_dispatch)" workflow_dispatch \
  -W "$ROOT/.github/workflows/ml-pipeline-trigger.yml" \
  -j ml-pipeline-trigger \
  --input pipeline_type=training

run "performance-regression-test (pull_request)" pull_request \
  -W "$ROOT/.github/workflows/performance-regression-test.yml" \
  -j performance-test \
  -e "$EVENT_PR"

echo ""
echo "All act jobs completed successfully."
echo ""
echo "Note: feature-release-artifact.yml is not run here (it targets branch release, uses the GitHub API, and may git push). Test it manually when needed."
