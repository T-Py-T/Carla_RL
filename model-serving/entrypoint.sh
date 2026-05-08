#!/bin/bash
# Production entrypoint for the CarlaRL Policy-as-a-Service.
#
# Uvicorn 0.24+ requires a lowercase --log-level value, while deployment
# pipelines frequently inject the env var in uppercase (INFO/DEBUG). We
# normalize it here before exec'ing uvicorn so either convention works.
set -e

LOG_LEVEL="$(echo "${LOG_LEVEL:-info}" | tr '[:upper:]' '[:lower:]')"

exec uvicorn src.server:app \
    --host 0.0.0.0 \
    --port "${PORT:-8080}" \
    --workers "${WORKERS:-1}" \
    --log-level "${LOG_LEVEL}" \
    --access-log \
    --no-use-colors
