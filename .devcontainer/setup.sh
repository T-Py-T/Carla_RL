#!/bin/bash
# Light post-create hook. The heavy dep install is baked into the image
# (see .devcontainer/Dockerfile), so this just verifies the environment
# looks right and regenerates the example model artifacts that are
# git-ignored (see model-serving/artifacts/README.md).

set -e

echo "[INFO] Carla_RL Dev Container ready."
echo "[INFO] Python: $(python --version 2>&1)"
echo "[INFO] Torch: $(python -c 'import torch; print(torch.__version__)' 2>&1)"
echo "[INFO] TensorFlow: $(python -c 'import tensorflow as tf; print(tf.__version__)' 2>&1)"
echo "[INFO] pytest: $(pytest --version 2>&1 | head -1)"

ARTIFACT_DIR=/workspace/model-serving/artifacts/v0.1.0
if [ ! -s "${ARTIFACT_DIR}/model.pt" ] || [ ! -s "${ARTIFACT_DIR}/preprocessor.pkl" ]; then
    echo "[INFO] Regenerating example artifacts in ${ARTIFACT_DIR} ..."
    (cd /workspace/model-serving \
        && python -m scripts.create_example_artifacts --output artifacts --version v0.1.0)
fi

echo ""
echo "[INFO] Testing recipes:"
echo "  cd /workspace/model-serving && pytest -q"
echo "  cd /workspace/model-sim     && pytest -q"
echo "  cd /workspace/model-serving && bash entrypoint.sh     # start FastAPI"
