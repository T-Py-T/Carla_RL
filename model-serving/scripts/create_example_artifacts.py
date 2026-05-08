#!/usr/bin/env python3
"""Create example model artifacts for CarlaRL Policy-as-a-Service.

Generates a small example policy and preprocessor so the serving stack has a
working `model.pt` + `preprocessor.pkl` pair to load. Intended for local
development, CI, and Docker image builds — not for production training.

Usage
-----
From the repo root so `src.*` imports resolve:

    python -m scripts.create_example_artifacts \
        --output model-serving/artifacts --version v0.1.0
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml

# Add the repo root to sys.path so `src.*` resolves when this script is run
# directly (e.g. `python scripts/create_example_artifacts.py`). When invoked
# with `-m scripts.create_example_artifacts`, the repo root is already on
# sys.path and the insert is a harmless no-op.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.io_schemas import Observation  # noqa: E402
from src.preprocessing import StandardFeaturePreprocessor  # noqa: E402


class ExampleCarlaModel(nn.Module):
    """Simple example model for CARLA autonomous driving.

    Input: 5 features (speed, steering, sensor1, sensor2, sensor3).
    Output: 3 floats that the serving post-processor clips into the
    [throttle, brake, steer] action space.
    """

    def __init__(
        self, input_dim: int = 5, hidden_dim: int = 64, output_dim: int = 3
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def create_example_model() -> nn.Module:
    """Create and lightly initialize the example model."""
    model = ExampleCarlaModel()
    with torch.no_grad():
        # Shrink the last layer so the untrained policy outputs small actions
        # and bias it towards gentle throttle + centered steering.
        model.network[4].weight.data *= 0.1
        model.network[4].bias.data = torch.tensor([0.5, -2.0, 0.0])
    return model


def create_example_preprocessor() -> StandardFeaturePreprocessor:
    """Create and fit a StandardFeaturePreprocessor on synthetic observations."""
    observations = [
        Observation(
            speed=20.0 + i * 0.5,
            steering=(i - 50) * 0.02,
            sensors=[
                0.5 + 0.01 * i,
                0.3 + 0.02 * i,
                0.7 - 0.01 * i,
            ],
        )
        for i in range(100)
    ]
    preprocessor = StandardFeaturePreprocessor(
        normalize_speed=True,
        normalize_steering=True,
        normalize_sensors=True,
        sensor_clip_range=(-10.0, 10.0),
    )
    preprocessor.fit(observations)
    return preprocessor


def compute_file_hash(file_path: Path) -> str:
    """Return the SHA-256 hex digest of `file_path`."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4096), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _save_model(model: nn.Module, model_path: Path) -> str:
    """Save `model` to `model_path`, preferring TorchScript.

    Tracing captures `forward` into a portable TorchScript module so loading
    does not require the original Python class on the import path. If tracing
    fails (e.g. on exotic custom ops), fall back to a full `nn.Module` pickle
    so the artifact still works via the PyTorch load path in model_loader.
    """
    model.eval()
    example_input = torch.randn(1, 5)
    try:
        scripted = torch.jit.trace(model, example_input)
        torch.jit.save(scripted, str(model_path))
        return "torchscript"
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: TorchScript tracing failed ({exc}); saving nn.Module pickle")
        torch.save(model, str(model_path))
        return "pytorch"


def create_artifacts(output_dir: Path, version: str = "v0.1.0") -> Path:
    """Create the complete artifact set under `output_dir/version`."""
    artifact_dir = output_dir / version
    artifact_dir.mkdir(parents=True, exist_ok=True)
    print(f"Creating example artifacts in {artifact_dir}")

    model_path = artifact_dir / "model.pt"
    save_format = _save_model(create_example_model(), model_path)
    print(f"  saved {save_format} model -> {model_path}")

    preprocessor_path = artifact_dir / "preprocessor.pkl"
    preprocessor = create_example_preprocessor()
    preprocessor.save(preprocessor_path)
    print(f"  saved preprocessor -> {preprocessor_path}")

    model_hash = compute_file_hash(model_path)
    preprocessor_hash = compute_file_hash(preprocessor_path)

    model_card_path = artifact_dir / "model_card.yaml"
    model_card = {}
    if model_card_path.exists():
        with open(model_card_path) as handle:
            model_card = yaml.safe_load(handle) or {}

    model_card.setdefault("model_name", "carla-ppo")
    model_card.setdefault("version", version)
    model_card.setdefault("model_type", save_format)
    model_card["artifact_hashes"] = {
        "model.pt": model_hash,
        "preprocessor.pkl": preprocessor_hash,
    }

    with open(model_card_path, "w") as handle:
        yaml.dump(model_card, handle, default_flow_style=False, sort_keys=False)
    print(f"  updated model card -> {model_card_path}")

    print("\nArtifacts created successfully. Files:")
    for entry in sorted(artifact_dir.iterdir()):
        if entry.is_file():
            size_mb = entry.stat().st_size / (1024 * 1024)
            print(f"  - {entry.name} ({size_mb:.2f} MB)")

    return artifact_dir


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create example CarlaRL model artifacts"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts"),
        help="Output directory for artifacts (relative paths resolved from CWD)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v0.1.0",
        help="Model version tag",
    )
    args = parser.parse_args(argv)

    try:
        artifact_dir = create_artifacts(args.output, args.version)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to create artifacts: {exc}")
        return 1

    print(f"\nDone. Artifacts ready at: {artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
