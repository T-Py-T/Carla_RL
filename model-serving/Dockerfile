# Multi-stage Dockerfile for CarlaRL Policy-as-a-Service
# Optimized for production deployment with security and performance

# Stage 1: Base image with system dependencies
FROM python:3.11-slim AS base

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Dependencies installation
FROM base AS dependencies

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Build example artifacts inside the image so freshly-cloned checkouts
# work without developers having to pre-generate `model.pt` / `preprocessor.pkl`.
FROM dependencies AS artifact-builder
WORKDIR /build
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY artifacts/ ./artifacts/
RUN python -m scripts.create_example_artifacts \
        --output artifacts \
        --version v0.1.0

# Stage 4: Production image
FROM python:3.11-slim AS production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    ARTIFACT_DIR="/app/artifacts" \
    PORT=8080 \
    WORKERS=1 \
    USE_GPU=0 \
    LOG_LEVEL=info

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r modelserving && useradd -r -g modelserving -s /bin/false modelserving

# Copy virtual environment from dependencies stage
COPY --from=dependencies /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/artifacts /app/logs \
    && chown -R modelserving:modelserving /app

# Copy application code, the freshly-generated artifact set, and the
# entrypoint script. The entrypoint normalizes LOG_LEVEL to lowercase so
# uvicorn accepts either INFO/info.
COPY --chown=modelserving:modelserving src/ ./src/
COPY --chown=modelserving:modelserving --from=artifact-builder /build/artifacts/ ./artifacts/
COPY --chown=modelserving:modelserving entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Switch to non-root user
USER modelserving

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/healthz || exit 1

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Labels for metadata
LABEL maintainer="CarlaRL Team" \
      version="1.0.0" \
      description="CarlaRL Policy-as-a-Service - High-performance RL policy serving" \
      org.opencontainers.image.source="https://github.com/your-org/carla-rl" \
      org.opencontainers.image.documentation="https://github.com/your-org/carla-rl/blob/main/README.md"
