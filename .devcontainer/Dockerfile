FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    ARTIFACT_DIR=/workspace/model-serving/artifacts \
    USE_GPU=0 \
    LOG_LEVEL=info

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN pip install --no-cache-dir uv

WORKDIR /workspace

COPY model-serving/requirements.txt /tmp/model-serving-requirements.txt
RUN pip install --no-cache-dir -r /tmp/model-serving-requirements.txt

RUN pip install --no-cache-dir \
        "highway-env>=1.10.2" \
        "gymnasium>=1.3.0" \
        "opencv-python-headless>=4.10.0" \
        "colorama>=0.4.6" \
        "wandb>=0.26.0" \
        "tensorflow>=2.21.0,<2.22" \
        "pytest-timeout>=2.4.0" \
        "ruff>=0.9.0" \
        "mypy>=1.18.0" \
        "pre-commit>=4.0.0"

CMD ["/bin/bash"]
