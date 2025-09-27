"""
Unit tests for model loading utilities in CarlaRL Policy-as-a-Service.

Tests model loading, artifact validation, and integrity checking.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
import yaml

from src.exceptions import ArtifactValidationError, ModelLoadingError
from src.model_loader import (
    PolicyWrapper,
    compute_file_hash,
    get_available_versions,
    load_artifacts,
    load_model_card,
    load_preprocessor,
    load_pytorch_model,
    validate_artifact_integrity,
    validate_model_compatibility,
)


class SimpleTestModel(nn.Module):
    """Simple test model for testing purposes."""

    def __init__(self, input_dim: int = 5, output_dim: int = 3):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.tanh(self.linear(x))


class TestPolicyWrapper:
    """Test cases for PolicyWrapper class."""

    def test_policy_wrapper_creation(self):
        """Test PolicyWrapper creation with PyTorch model."""
        model = SimpleTestModel()
        wrapper = PolicyWrapper(model, model_type="pytorch")

        assert wrapper.model_type == "pytorch"
        assert isinstance(wrapper.device, torch.device)

    def test_policy_wrapper_forward(self):
        """Test forward pass through PolicyWrapper."""
        model = SimpleTestModel(input_dim=5, output_dim=3)
        wrapper = PolicyWrapper(model)

        x = torch.randn(2, 5)  # batch_size=2, input_dim=5

        # Test forward pass
        output = wrapper(x, deterministic=True)
        assert output.shape == (2, 3)
        assert torch.allclose(output, torch.tanh(model.linear(x)))

    def test_policy_wrapper_device_management(self):
        """Test device management in PolicyWrapper."""
        model = SimpleTestModel()
        wrapper = PolicyWrapper(model)

        original_device = wrapper.device

        # Test moving to same device (should work)
        wrapper_moved = wrapper.to(original_device)
        assert wrapper_moved.device == original_device

    def test_policy_wrapper_torchscript(self):
        """Test PolicyWrapper with TorchScript model."""
        model = SimpleTestModel()
        scripted_model = torch.jit.script(model)
        wrapper = PolicyWrapper(scripted_model, model_type="torchscript")

        assert wrapper.model_type == "torchscript"

        x = torch.randn(1, 5)
        output = wrapper(x)
        assert output.shape == (1, 3)

    def test_policy_wrapper_forward_error(self):
        """Test error handling in forward pass."""
        # Create a model that will fail
        model = Mock()
        model.parameters.return_value = iter([torch.tensor([1.0])])
        model.eval.return_value = model
        model.side_effect = RuntimeError("Model error")

        wrapper = PolicyWrapper(model)

        with pytest.raises(ModelLoadingError) as exc_info:
            wrapper(torch.randn(1, 5))

        assert "Forward pass failed" in str(exc_info.value)


class TestFileOperations:
    """Test cases for file operations."""

    def test_compute_file_hash(self):
        """Test file hash computation."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            temp_path = Path(f.name)

        try:
            hash_value = compute_file_hash(temp_path)

            # Verify it's a valid SHA256 hash
            assert len(hash_value) == 64
            assert all(c in "0123456789abcdef" for c in hash_value)

            # Verify consistency
            hash_value2 = compute_file_hash(temp_path)
            assert hash_value == hash_value2

        finally:
            temp_path.unlink()

    def test_compute_file_hash_nonexistent(self):
        """Test hash computation for nonexistent file."""
        with pytest.raises(ArtifactValidationError):
            compute_file_hash(Path("nonexistent_file.txt"))


class TestModelCardOperations:
    """Test cases for model card operations."""

    def create_test_model_card(self, temp_dir: Path, **kwargs) -> Path:
        """Create a test model card file."""
        model_card_data = {
            "model_name": "test-model",
            "version": "v0.1.0",
            "model_type": "pytorch",
            "input_shape": [5],
            "output_shape": [3],
            "framework_version": "2.1.0",
            **kwargs,
        }

        model_card_path = temp_dir / "model_card.yaml"
        with open(model_card_path, "w") as f:
            yaml.dump(model_card_data, f)

        return model_card_path

    def test_load_model_card_success(self):
        """Test successful model card loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            self.create_test_model_card(temp_path)

            model_card = load_model_card(temp_path)

            assert model_card["model_name"] == "test-model"
            assert model_card["version"] == "v0.1.0"
            assert model_card["model_type"] == "pytorch"

    def test_load_model_card_missing_file(self):
        """Test model card loading with missing file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with pytest.raises(ModelLoadingError) as exc_info:
                load_model_card(temp_path)

            assert "Model card not found" in str(exc_info.value)

    def test_load_model_card_missing_fields(self):
        """Test model card loading with missing required fields."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create model card missing required fields
            model_card_data = {"model_name": "test-model"}  # Missing version and model_type
            model_card_path = temp_path / "model_card.yaml"
            with open(model_card_path, "w") as f:
                yaml.dump(model_card_data, f)

            with pytest.raises(ModelLoadingError) as exc_info:
                load_model_card(temp_path)

            assert "missing required fields" in str(exc_info.value)

    def test_load_model_card_invalid_yaml(self):
        """Test model card loading with invalid YAML."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create invalid YAML file
            model_card_path = temp_path / "model_card.yaml"
            with open(model_card_path, "w") as f:
                f.write("invalid: yaml: content: [unclosed")

            with pytest.raises(ModelLoadingError) as exc_info:
                load_model_card(temp_path)

            assert "Failed to parse model card YAML" in str(exc_info.value)


class TestArtifactValidation:
    """Test cases for artifact validation."""

    def test_validate_artifact_integrity_success(self):
        """Test successful artifact integrity validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            test_file = temp_path / "test.txt"
            test_file.write_text("test content")

            # Compute actual hash
            actual_hash = compute_file_hash(test_file)

            # Create model card with correct hash
            model_card = {"artifact_hashes": {"test.txt": actual_hash}}

            # Should pass validation
            result = validate_artifact_integrity(temp_path, model_card)
            assert result is True

    def test_validate_artifact_integrity_hash_mismatch(self):
        """Test artifact validation with hash mismatch."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test file
            test_file = temp_path / "test.txt"
            test_file.write_text("test content")

            # Create model card with wrong hash
            model_card = {"artifact_hashes": {"test.txt": "wrong_hash"}}

            with pytest.raises(ArtifactValidationError) as exc_info:
                validate_artifact_integrity(temp_path, model_card)

            assert "Hash mismatch" in str(exc_info.value)

    def test_validate_artifact_integrity_missing_file(self):
        """Test artifact validation with missing file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create model card referencing nonexistent file
            model_card = {"artifact_hashes": {"missing.txt": "some_hash"}}

            with pytest.raises(ArtifactValidationError) as exc_info:
                validate_artifact_integrity(temp_path, model_card)

            assert "Missing artifact file" in str(exc_info.value)

    def test_validate_artifact_integrity_no_hashes(self):
        """Test artifact validation with no hashes in model card."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Model card without artifact_hashes
            model_card = {"model_name": "test"}

            # Should pass (no validation needed)
            result = validate_artifact_integrity(temp_path, model_card)
            assert result is True


class TestModelLoading:
    """Test cases for model loading."""

    def create_test_model_file(self, temp_dir: Path, filename: str = "model.pt") -> Path:
        """Create a test model file."""
        model = SimpleTestModel()
        model_path = temp_dir / filename
        torch.save(model, model_path)
        return model_path

    def test_load_pytorch_model_success(self):
        """Test successful PyTorch model loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_path = self.create_test_model_file(temp_path)

            wrapper = load_pytorch_model(model_path, torch.device("cpu"))

            assert isinstance(wrapper, PolicyWrapper)
            assert wrapper.model_type == "pytorch"

    def test_load_pytorch_model_torchscript(self):
        """Test loading TorchScript model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create and save TorchScript model
            model = SimpleTestModel()
            scripted_model = torch.jit.script(model)
            model_path = temp_path / "model.pt"
            torch.jit.save(scripted_model, model_path)

            wrapper = load_pytorch_model(model_path, torch.device("cpu"))

            assert isinstance(wrapper, PolicyWrapper)
            assert wrapper.model_type == "torchscript"

    def test_load_pytorch_model_nonexistent(self):
        """Test loading nonexistent model file."""
        with pytest.raises(ModelLoadingError):
            load_pytorch_model(Path("nonexistent.pt"), torch.device("cpu"))

    def test_load_preprocessor_success(self):
        """Test successful preprocessor loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test preprocessor
            test_preprocessor = {"type": "test", "fitted": True}
            preprocessor_path = temp_path / "preprocessor.pkl"

            import pickle

            with open(preprocessor_path, "wb") as f:
                pickle.dump(test_preprocessor, f)

            loaded_preprocessor = load_preprocessor(preprocessor_path)
            assert loaded_preprocessor == test_preprocessor

    def test_load_preprocessor_missing(self):
        """Test loading missing preprocessor."""
        result = load_preprocessor(Path("nonexistent.pkl"))
        assert result is None

    def test_load_preprocessor_corrupted(self):
        """Test loading corrupted preprocessor."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create corrupted file
            preprocessor_path = temp_path / "preprocessor.pkl"
            preprocessor_path.write_text("corrupted content")

            with pytest.raises(ModelLoadingError):
                load_preprocessor(preprocessor_path)


class TestArtifactLoading:
    """Test cases for complete artifact loading."""

    def create_test_artifacts(self, temp_dir: Path) -> None:
        """Create complete test artifacts."""
        # Create model
        model = SimpleTestModel()
        model_path = temp_dir / "model.pt"
        torch.save(model, model_path)

        # Create preprocessor
        preprocessor = {"type": "test", "fitted": True}
        preprocessor_path = temp_dir / "preprocessor.pkl"
        import pickle

        with open(preprocessor_path, "wb") as f:
            pickle.dump(preprocessor, f)

        # Create model card
        model_card_data = {
            "model_name": "test-model",
            "version": "v0.1.0",
            "model_type": "pytorch",
            "input_shape": [5],
            "output_shape": [3],
            "framework_version": "2.1.0",
        }
        model_card_path = temp_dir / "model_card.yaml"
        with open(model_card_path, "w") as f:
            yaml.dump(model_card_data, f)

    def test_load_artifacts_success(self):
        """Test successful artifact loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            self.create_test_artifacts(temp_path)

            policy, preprocessor = load_artifacts(
                temp_path, torch.device("cpu"), validate_integrity=False
            )

            assert isinstance(policy, PolicyWrapper)
            assert preprocessor is not None

    def test_load_artifacts_missing_directory(self):
        """Test artifact loading with missing directory."""
        with pytest.raises(ModelLoadingError) as exc_info:
            load_artifacts(Path("nonexistent"), torch.device("cpu"))

        assert "Artifact directory not found" in str(exc_info.value)

    def test_load_artifacts_missing_model(self):
        """Test artifact loading with missing model file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create only model card, no model file
            model_card_data = {
                "model_name": "test-model",
                "version": "v0.1.0",
                "model_type": "pytorch",
            }
            model_card_path = temp_path / "model_card.yaml"
            with open(model_card_path, "w") as f:
                yaml.dump(model_card_data, f)

            with pytest.raises(ModelLoadingError) as exc_info:
                load_artifacts(temp_path, torch.device("cpu"))

            assert "Model file not found" in str(exc_info.value)


class TestVersionManagement:
    """Test cases for version management."""

    def test_get_available_versions(self):
        """Test getting available model versions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create version directories
            (temp_path / "v0.1.0").mkdir()
            (temp_path / "v0.2.0").mkdir()
            (temp_path / "v1.0.0").mkdir()
            (temp_path / "not_a_version").mkdir()  # Should be ignored

            versions = get_available_versions(temp_path)

            assert "v0.1.0" in versions
            assert "v0.2.0" in versions
            assert "v1.0.0" in versions
            assert "not_a_version" not in versions

    def test_get_available_versions_empty(self):
        """Test getting versions from empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            versions = get_available_versions(temp_path)
            assert versions == []

    def test_get_available_versions_nonexistent(self):
        """Test getting versions from nonexistent directory."""
        versions = get_available_versions(Path("nonexistent"))
        assert versions == []


class TestModelCompatibility:
    """Test cases for model compatibility validation."""

    def test_validate_model_compatibility_success(self):
        """Test successful model compatibility validation."""
        model_card = {"input_shape": [5], "output_shape": [3], "framework_version": "2.1.0"}

        result = validate_model_compatibility(model_card)
        assert result is True

    def test_validate_model_compatibility_missing_fields(self):
        """Test compatibility validation with missing fields."""
        model_card = {"input_shape": [5]}  # Missing output_shape and framework_version

        with pytest.raises(ArtifactValidationError) as exc_info:
            validate_model_compatibility(model_card)

        assert "missing compatibility fields" in str(exc_info.value)

    def test_validate_model_compatibility_invalid_shapes(self):
        """Test compatibility validation with invalid shapes."""
        # Invalid input_shape
        model_card = {
            "input_shape": "not_a_list",
            "output_shape": [3],
            "framework_version": "2.1.0",
        }

        with pytest.raises(ArtifactValidationError) as exc_info:
            validate_model_compatibility(model_card)

        assert "Invalid input_shape" in str(exc_info.value)
