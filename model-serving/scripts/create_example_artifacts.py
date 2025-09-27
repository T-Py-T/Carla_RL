#!/usr/bin/env python3
"""
Script to create example model artifacts for CarlaRL Policy-as-a-Service.

This script generates a simple example model and preprocessor for testing
and demonstration purposes.
"""

import argparse
import hashlib

# Add src to path for imports
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).parent.parent / "src"))

from io_schemas import Observation
from preprocessing import StandardFeaturePreprocessor


class ExampleCarlaModel(nn.Module):
    """Simple example model for CARLA autonomous driving."""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, output_dim: int = 3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Output in [-1, 1] range
        )

    def forward(self, x):
        return self.network(x)

    def act(self, x, deterministic=False):
        """Policy action method for RL inference."""
        output = self.forward(x)

        # Convert to action space: [throttle, brake, steer]
        # Throttle and brake: [0, 1], Steer: [-1, 1]
        throttle = torch.sigmoid(output[:, 0])  # [0, 1]
        brake = torch.sigmoid(output[:, 1])     # [0, 1]
        steer = output[:, 2]                    # [-1, 1] (already from tanh)

        return torch.stack([throttle, brake, steer], dim=1)


def create_example_model() -> nn.Module:
    """Create and initialize example model."""
    model = ExampleCarlaModel()

    # Initialize with reasonable weights for driving
    with torch.no_grad():
        # Make the model prefer moderate throttle, low brake, and centered steering
        model.network[4].weight.data *= 0.1  # Reduce output magnitude
        model.network[4].bias.data = torch.tensor([0.5, -2.0, 0.0])  # Bias towards throttle, away from brake

    return model


def create_example_preprocessor() -> StandardFeaturePreprocessor:
    """Create and fit example preprocessor."""
    # Create example training observations
    example_observations = []
    for i in range(100):
        obs = Observation(
            speed=20.0 + i * 0.5,  # Speeds from 20 to 70 km/h
            steering=(i - 50) * 0.02,  # Steering from -1.0 to 1.0
            sensors=[
                0.5 + 0.01 * i,  # Sensor 1
                0.3 + 0.02 * i,  # Sensor 2
                0.7 - 0.01 * i,  # Sensor 3
            ]
        )
        example_observations.append(obs)

    # Create and fit preprocessor
    preprocessor = StandardFeaturePreprocessor(
        normalize_speed=True,
        normalize_steering=True,
        normalize_sensors=True,
        sensor_clip_range=(-10.0, 10.0)
    )

    preprocessor.fit(example_observations)
    return preprocessor


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def create_artifacts(output_dir: Path, version: str = "v0.1.0"):
    """Create complete artifact set."""
    print(f"Creating example artifacts in {output_dir}")

    # Create output directory
    artifact_dir = output_dir / version
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Create model
    print("Creating example model...")
    model = create_example_model()
    model.eval()

    # Save as TorchScript for production
    example_input = torch.randn(1, 5)
    try:
        scripted_model = torch.jit.trace(model, example_input)
        model_path = artifact_dir / "model.pt"
        torch.jit.save(scripted_model, model_path)
        print(f"✅ Saved TorchScript model to {model_path}")
    except Exception as e:
        print(f"⚠️  TorchScript failed ({e}), saving regular PyTorch model")
        model_path = artifact_dir / "model.pt"
        torch.save(model, model_path)
        print(f"✅ Saved PyTorch model to {model_path}")

    # Create preprocessor
    print("Creating example preprocessor...")
    preprocessor = create_example_preprocessor()
    preprocessor_path = artifact_dir / "preprocessor.pkl"
    preprocessor.save(preprocessor_path)
    print(f"✅ Saved preprocessor to {preprocessor_path}")

    # Compute hashes
    model_hash = compute_file_hash(model_path)
    preprocessor_hash = compute_file_hash(preprocessor_path)

    print(f"Model hash: {model_hash}")
    print(f"Preprocessor hash: {preprocessor_hash}")

    # Update model card with computed hashes
    model_card_path = artifact_dir / "model_card.yaml"
    if model_card_path.exists():
        import yaml
        with open(model_card_path) as f:
            model_card = yaml.safe_load(f)

        model_card["artifact_hashes"] = {
            "model.pt": model_hash,
            "preprocessor.pkl": preprocessor_hash
        }

        with open(model_card_path, 'w') as f:
            yaml.dump(model_card, f, default_flow_style=False)

        print(f"✅ Updated model card with hashes at {model_card_path}")
    else:
        print("⚠️  Model card not found, hashes not updated")

    print("\n🎉 Example artifacts created successfully!")
    print(f"Artifact directory: {artifact_dir}")
    print("Files created:")
    for file_path in artifact_dir.iterdir():
        if file_path.is_file():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  - {file_path.name} ({size_mb:.2f} MB)")

    return artifact_dir


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Create example CarlaRL model artifacts")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts"),
        help="Output directory for artifacts"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v0.1.0",
        help="Model version"
    )

    args = parser.parse_args()

    try:
        artifact_dir = create_artifacts(args.output, args.version)
        print(f"\n✅ Success! Artifacts ready at: {artifact_dir}")
        print("\nTo test the artifacts:")
        print("  make dev")
        print("  curl http://localhost:8080/healthz")

    except Exception as e:
        print(f"\n❌ Failed to create artifacts: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
