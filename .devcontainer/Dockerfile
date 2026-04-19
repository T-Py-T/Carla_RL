FROM python:3.11-slim

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
        "numpy==1.26.2" \
        "highway-env>=1.10.1" \
        "gymnasium>=1.1.1" \
        "opencv-python-headless>=4.5.0" \
        "colorama>=0.4.4" \
        "wandb>=0.22.0" \
        "tensorflow>=2.21.0,<2.22" \
        "pytest-timeout>=2.2.0" \
        "ruff>=0.1.8" \
        "mypy>=1.0.0" \
        "pre-commit>=3.0.0"

CMD ["/bin/bash"]
