"""
Model loading utilities for CarlaRL Policy-as-a-Service.

This module handles loading of model artifacts including TorchScript models,
ONNX models, and associated metadata with integrity validation.
"""

import hashlib
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
import yaml
from torch import jit

from .exceptions import ArtifactValidationError, ModelLoadingError
from .preprocessing import FeaturePreprocessor


class PolicyWrapper(nn.Module):
    """
    Wrapper for RL policy models with deterministic/stochastic inference modes.

    Provides a unified interface for different model formats and inference modes.
    """

    def __init__(self, model: nn.Module | jit.ScriptModule, model_type: str = "pytorch"):
        super().__init__()
        self.model = model.eval()
        self.model_type = model_type

        # Safely get device, handling cases where model has no parameters
        try:
            if hasattr(model, "parameters"):
                # Try to get device from first parameter
                first_param = next(iter(model.parameters()), None)
                self._device = (
                    first_param.device if first_param is not None else torch.device("cpu")
                )
            else:
                self._device = torch.device("cpu")
        except (StopIteration, AttributeError):
            # Fallback to CPU if no parameters or other issues
            self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return self._device

    def to(self, *args: Any, **kwargs: Any) -> "PolicyWrapper":
        """Move model to specified device."""
        self.model = self.model.to(*args, **kwargs)
        self._device = next(iter(self.model.parameters()), torch.empty(0)).device
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Forward pass through the policy model.

        Args:
            x: Input tensor (batch_size, input_dim)
            deterministic: Whether to use deterministic inference

        Returns:
            Action tensor (batch_size, action_dim)
        """
        try:
            # TorchScript modules expose a fixed forward signature captured at
            # trace/script time. Probing with deterministic=True first and
            # falling back to a single-argument call keeps us compatible with
            # both stochastic policies (that accept the flag) and simple
            # traced modules (that only accept the input tensor).
            if self.model_type == "torchscript":
                act = getattr(self.model, "act", None)
                if callable(act):
                    try:
                        return cast(torch.Tensor, act(x, deterministic))
                    except (TypeError, RuntimeError):
                        return cast(torch.Tensor, act(x))
                try:
                    return cast(torch.Tensor, self.model(x, deterministic))
                except (TypeError, RuntimeError):
                    return cast(torch.Tensor, self.model(x))

            act = getattr(self.model, "act", None)
            if callable(act):
                return cast(torch.Tensor, act(x, deterministic))
            return cast(torch.Tensor, self.model(x))

        except Exception as e:
            raise ModelLoadingError(
                f"Forward pass failed: {str(e)}",
                details={
                    "model_type": self.model_type,
                    "input_shape": list(x.shape),
                    "deterministic": deterministic,
                },
            )


def compute_file_hash(file_path: Path) -> str:
    """
    Compute SHA256 hash of a file for integrity validation.

    Args:
        file_path: Path to file

    Returns:
        Hexadecimal hash string
    """
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except Exception as e:
        raise ArtifactValidationError(f"Failed to compute hash for {file_path}: {str(e)}")


def validate_artifact_integrity(artifact_dir: Path, model_card: dict[str, Any]) -> None:
    """
    Validate artifact integrity using hashes from model card.

    Args:
        artifact_dir: Directory containing artifacts
        model_card: Model card metadata with expected hashes

    Returns:
        None. Invalid artifacts raise ``ArtifactValidationError``.

    Raises:
        ArtifactValidationError: If validation fails
    """
    expected_hashes = model_card.get("artifact_hashes") or {}
    if not expected_hashes:
        # No hashes recorded in the model card - skip validation. This is the
        # expected path for freshly generated artifacts before
        # create_example_artifacts.py has populated the hash map.
        return

    for filename, expected_hash in expected_hashes.items():
        file_path = artifact_dir / filename

        if not file_path.exists():
            raise ArtifactValidationError(
                f"Missing artifact file: {filename}",
                details={"expected_files": list(expected_hashes.keys())},
            )

        actual_hash = compute_file_hash(file_path)
        if actual_hash != expected_hash:
            raise ArtifactValidationError(
                f"Hash mismatch for {filename}",
                details={
                    "expected_hash": expected_hash,
                    "actual_hash": actual_hash,
                    "file_path": str(file_path),
                },
            )


def load_model_card(artifact_dir: Path) -> dict[str, Any]:
    """
    Load and parse model card YAML file.

    Args:
        artifact_dir: Directory containing model_card.yaml

    Returns:
        Model card metadata dictionary

    Raises:
        ModelLoadingError: If model card cannot be loaded
    """
    model_card_path = artifact_dir / "model_card.yaml"

    if not model_card_path.exists():
        raise ModelLoadingError(
            f"Model card not found: {model_card_path}",
            details={"artifact_dir": str(artifact_dir)},
        )

    try:
        with open(model_card_path) as f:
            model_card = yaml.safe_load(f)

        if not isinstance(model_card, dict) or not all(isinstance(key, str) for key in model_card):
            raise ModelLoadingError(
                "Model card root must be an object with string keys",
                details={"model_card_path": str(model_card_path)},
            )

        # Validate required fields
        required_fields = ["model_name", "version", "model_type"]
        missing_fields = [field for field in required_fields if field not in model_card]

        if missing_fields:
            raise ModelLoadingError(
                f"Model card missing required fields: {missing_fields}",
                details={"model_card_path": str(model_card_path)},
            )

        return cast(dict[str, Any], model_card)

    except yaml.YAMLError as e:
        raise ModelLoadingError(
            f"Failed to parse model card YAML: {str(e)}",
            details={"model_card_path": str(model_card_path)},
        )
    except Exception as e:
        raise ModelLoadingError(
            f"Failed to load model card: {str(e)}",
            details={"model_card_path": str(model_card_path)},
        )


def load_pytorch_model(model_path: Path, device: torch.device) -> PolicyWrapper:
    """
    Load PyTorch model from file.

    Args:
        model_path: Path to model file
        device: Target device for model

    Returns:
        Wrapped policy model

    Raises:
        ModelLoadingError: If model cannot be loaded
    """
    try:
        # Try loading as TorchScript first
        model = torch.jit.load(str(model_path), map_location=device)
        return PolicyWrapper(model, model_type="torchscript")

    except Exception as torchscript_error:
        try:
            # Inspect data-only checkpoints without allowing arbitrary Python
            # object construction. State dictionaries still require an
            # application-owned architecture registry, which this loader does
            # not have, so reject them with a useful diagnostic.
            checkpoint = torch.load(str(model_path), map_location=device, weights_only=True)
            available_keys = list(checkpoint.keys()) if isinstance(checkpoint, dict) else []
            raise ModelLoadingError(
                "Safe PyTorch checkpoint loading requires a registered model architecture; "
                "export this model as TorchScript",
                details={"available_keys": available_keys},
            )

        except Exception as pytorch_error:
            raise ModelLoadingError(
                f"Failed to load model from {model_path}",
                details={
                    "torchscript_error": str(torchscript_error),
                    "pytorch_error": str(pytorch_error),
                },
            )


def load_preprocessor(preprocessor_path: Path) -> FeaturePreprocessor | None:
    """
    Load a data-only JSON preprocessor artifact.

    Args:
        preprocessor_path: Path to the serialized preprocessor file

    Returns:
        Loaded preprocessor object or None if file doesn't exist

    Raises:
        ModelLoadingError: If preprocessor cannot be loaded
    """
    if not preprocessor_path.exists():
        return None

    try:
        return FeaturePreprocessor.load(preprocessor_path)

    except Exception as e:
        raise ModelLoadingError(
            f"Failed to load preprocessor from {preprocessor_path}: {str(e)}",
            details={"preprocessor_path": str(preprocessor_path)},
        )


def load_artifacts(
    artifact_dir: Path, device: torch.device, validate_integrity: bool = True
) -> tuple[PolicyWrapper, FeaturePreprocessor | None]:
    """
    Load model artifacts from directory.

    Args:
        artifact_dir: Directory containing model artifacts
        device: Target device for model
        validate_integrity: Whether to validate artifact integrity

    Returns:
        Tuple of (wrapped_policy_model, preprocessor)

    Raises:
        ModelLoadingError: If artifacts cannot be loaded
        ArtifactValidationError: If integrity validation fails
    """
    if not artifact_dir.exists():
        raise ModelLoadingError(
            f"Artifact directory not found: {artifact_dir}",
            details={"artifact_dir": str(artifact_dir)},
        )

    # Load model card first
    model_card = load_model_card(artifact_dir)

    # Validate artifact integrity if requested
    if validate_integrity:
        validate_artifact_integrity(artifact_dir, model_card)

    # Determine model file path
    model_filename = model_card.get("model_filename", "model.pt")
    model_path = artifact_dir / model_filename

    if not model_path.exists():
        raise ModelLoadingError(
            f"Model file not found: {model_path}",
            details={
                "artifact_dir": str(artifact_dir),
                "expected_filename": model_filename,
            },
        )

    # Load model based on type
    model_type = model_card.get("model_type", "pytorch").lower()

    if model_type in ["pytorch", "torchscript"]:
        policy = load_pytorch_model(model_path, device)
    else:
        raise ModelLoadingError(
            f"Unsupported model type: {model_type}",
            details={"supported_types": ["pytorch", "torchscript"]},
        )

    # Load preprocessor if available
    preprocessor_filename = model_card.get("preprocessor_filename")
    if preprocessor_filename:
        preprocessor_path = artifact_dir / preprocessor_filename
    else:
        json_path = artifact_dir / "preprocessor.json"
        preprocessor_path = json_path if json_path.exists() else artifact_dir / "preprocessor.pkl"
    preprocessor = load_preprocessor(preprocessor_path)

    return policy, preprocessor


def get_available_versions(artifacts_root: Path) -> list[str]:
    """
    Get list of available model versions.

    Args:
        artifacts_root: Root directory containing version subdirectories

    Returns:
        List of version strings (e.g., ['v0.1.0', 'v0.2.0'])
    """
    if not artifacts_root.exists():
        return []

    versions = []
    for item in artifacts_root.iterdir():
        if item.is_dir() and item.name.startswith("v"):
            # Basic semantic version validation
            if len(item.name.split(".")) >= 3:
                versions.append(item.name)

    # Sort versions (basic lexicographic sorting)
    return sorted(versions, reverse=True)


def validate_model_compatibility(model_card: dict[str, Any]) -> bool:
    """
    Validate model compatibility with current serving infrastructure.

    Args:
        model_card: Model card metadata

    Returns:
        True if model is compatible

    Raises:
        ArtifactValidationError: If model is incompatible
    """
    # Check required fields
    required_fields = ["input_shape", "output_shape", "framework_version"]
    missing_fields = [field for field in required_fields if field not in model_card]

    if missing_fields:
        raise ArtifactValidationError(f"Model card missing compatibility fields: {missing_fields}")

    # Validate input/output shapes
    input_shape = model_card["input_shape"]
    output_shape = model_card["output_shape"]

    if not isinstance(input_shape, list) or len(input_shape) == 0:
        raise ArtifactValidationError("Invalid input_shape in model card")

    if not isinstance(output_shape, list) or len(output_shape) == 0:
        raise ArtifactValidationError("Invalid output_shape in model card")

    # Basic framework version check (could be more sophisticated)
    framework_version = model_card["framework_version"]
    if not isinstance(framework_version, str):
        raise ArtifactValidationError("Invalid framework_version in model card")

    return True
